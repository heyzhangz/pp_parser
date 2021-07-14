import nltk
import json
from nltk.tokenize import WordPunctTokenizer
from nltk.parse import corenlp
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

PERM_KEYWORD_LIST = ["contact", "camera"]

def getPos(postag):

    res = \
        wordnet.ADJ  if postag.startswith('J') else \
        wordnet.VERB if postag.startswith('V') else \
        wordnet.NOUN if postag.startswith('N') else \
        wordnet.ADV  if postag.startswith('R') else \
        ""

    return res

class DepParser():

    def __init__(self):

        super().__init__()
        self.stopwordList = stopwords.words("english")
        self.depParser = corenlp.CoreNLPDependencyParser(url=r"http://localhost:9000")
        self.senTokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        self.lemmatizer = WordNetLemmatizer()

        pass

    def parse(self, sentence: str):

        # ,= 这边是迭代器只有一个值，所以只取一个
        parseRes, = self.depParser.raw_parse(sentence)
        parseRes = parseRes.to_conll(4)

        res = [{
            "dependent": "ROOT",
            "pos": "ROOT",
            "govloc": 0,
            "dep": "ROOT"
        }]
        for worddep in parseRes.split('\n'):
            
            worddep = worddep.strip()
            if "" == worddep:
                continue
            
            dependent, pos, governorloc, dep = worddep.split('\t')
            try:
                dependent = self.lemmatizer.lemmatize(dependent.lower(), getPos(pos))
            except KeyError:
                dependent = self.lemmatizer.lemmatize(dependent.lower())

            res.append({
                "dependent": dependent,
                "pos": pos,
                "govloc": int(governorloc),
                "dep": dep
            })

        return res

    def prettyRes(self, usrinput):

        if type(usrinput) == str:
            usrinput = self.parse(usrinput)
        
        resstr = ""
        for idx, item in enumerate(usrinput):
            dependent = item["dependent"]
            pos = item["pos"]
            governorloc = item["govloc"]
            dep = item["dep"]
            governor = usrinput[governorloc]["dependent"]
            governorpos = usrinput[governorloc]["pos"]

            resstr += ("%d\t%s%s%s\n" % (idx, 
                      addSpaces("(%s, %s)" % (dependent, pos), 24),
                      addSpaces(dep, 16),
                      addSpaces("[%d](%s, %s)" % (governorloc, governor, governorpos), 28)))

        
        return resstr
    pass

def addSpaces(s, length):
    
    if len(s) < length:
        s += " " * (length - len(s))

    return s

def isCompound(relation):
    """
       组合关系, 相邻两个词可以组合为短语.
       目前将两个类型算作组合:
       1. compound(and its subtype)
       2. amod
    """
    if relation.startswith("compound"):
        return True
    
    if relation == "amod":
        return True

    return False

def isInvalidPos(pos):
    # TODO 暂时只关注四个词性，后面有问题再改
    if getPos(pos) == "":
        return True

    return False

class SentenceParser():

    def __init__(self):

        super().__init__()
        self.depParser = DepParser()

        # TODO for debug
        self.depRes = None

        pass

    def _formatDepRes(self, depRes):
        """
            预处理依存关系结果:
            1. 合并全部**相邻**compound/amod关系, 包括sub-compund
        """

        if not depRes:
            print("[ERROR] The dependency result is None.")
            return None

        # 从后往前找全部相邻的compound合并
        locMap = [x for x in range(len(depRes))] # 更新后的位置映射关系

        def findTrueGov(loc):
            if loc >= len(locMap):
                return -1
            # 找并查集真正的相邻根节点
            while loc != locMap[loc]:
                loc = locMap[loc]
            
            return loc

        for idx, item in reversed(list(enumerate(depRes))):
            dependent = item["dependent"]
            dep = item["dep"]
            gloc = findTrueGov(item["govloc"])
            siblinggloc = findTrueGov(idx + 1)

            if isCompound(dep) and siblinggloc == gloc:
                # 记录 dep loc 到 gov loc 的映射
                locMap[idx] = gloc
                # 更新gov
                depRes[gloc]["dependent"] = "%s %s" % (dependent, depRes[gloc]["dependent"])
        
        # 更新全部loc信息
        delta = 0
        for idx, gloc in enumerate(locMap):
            
            if idx != gloc:
                # 删除节点
                del(depRes[idx - delta])
                delta += 1    
            locMap[idx] = delta
        
        for idx, item in enumerate(depRes):
            depRes[idx]["govloc"] -= locMap[depRes[idx]["govloc"]]
        
        return depRes

    def _findWordLocs(self, keywords, depRes):

        res = []

        if type(keywords) == str:
            keywords = [keywords]

        for idx, item in enumerate(depRes):
            word = item["dependent"]
            # TODO 后面可以换相似度之类的方法, 现在就只是匹配一哈
            for key in keywords:
                if key in word:
                    res.append(idx)
                    break
        return res

    def _parseFinVerb(self, nloc, depRes):
        
        govloc = depRes[nloc]["govloc"]
        govpos = depRes[govloc]["pos"]
        dep = depRes[govloc]["dep"]
        # dep == "obj" case: "The app needs access to the camera to fulfill recording videos."
        while getPos(govpos) == wordnet.VERB or dep == "obj":
            nloc = govloc
            govloc = depRes[nloc]["govloc"]
            govpos = depRes[govloc]["pos"]
            dep = depRes[govloc]["dep"]
            
        if getPos(depRes[nloc]["pos"]) != wordnet.VERB:
            return -1

        return nloc

    def _parseDepWord(self, wloc, depRes):
        
        dlocs = []

        for idx, item in enumerate(depRes):
            govloc = item["govloc"]
            if govloc == wloc:
                dlocs.append(idx)

        return dlocs

    def _findPhraseEnd(self, loc, depRes):
        """
            根据某个词查找完整短语结束位置, 目前暂时考虑使用这个词前后到最近governor位置做截断
        """

        finloc = loc

        for dist in range(1, len(depRes)):

            left = loc - dist
            right = loc + dist

            if left >= 0:
                govloc = depRes[left]["govloc"]
                if govloc == loc and not isInvalidPos(depRes[left]["pos"]):
                    finloc = left
            
            if right < len(depRes):
                govloc = depRes[right]["govloc"]
                if govloc == loc and not isInvalidPos(depRes[right]["pos"]):
                    finloc = right
            
            if finloc != loc:
                break

        return finloc

    def _getPhrase(self, start, end, depRes):
        
        phrase = []

        if start > end:
            start, end = end, start

        for idx in range(start, end + 1):
            phrase.append(depRes[idx]["dependent"])
        
        return ' '.join(phrase)

    def _pattern1(self, keyloc, depRes):
        """
            case: Images recorded by cameras fitted to Sky's engineer vans.
            4 (cameras, NNS) obl [2](recorded, VBN)

            利用PI直接找到场景动词
            
            条件: PI的依赖关系为 obl
        """

        res = []
        if depRes[keyloc]["dep"] != "obl":
            return res

        deploc = depRes[keyloc]["govloc"]
        finloc = self._findPhraseEnd(deploc, depRes)
        phrase = self._getPhrase(deploc, finloc, depRes)

        res.append([depRes[keyloc]["dependent"], phrase, depRes[keyloc]["dep"], depRes[keyloc]["dependent"], "pattern_1"])

        return res
    
    def _pattern2(self, keyloc, depRes):
        """
            case: Permission to access contact information is used when you search contacts in JVSTUDIOS search bar.
            4  (contact, NN) compound [5](information, NN)
            10 (search, VBP) advcl    [7](used, VBN)

            case: The app needs access to the camera to fulfill recording videos.
            7 (camera, NN)   nmod  [4](access, NN)
            9 (fulfill, VB)  xcomp [3](needs, VBZ)

            利用PI找到依赖动词, 利用依赖动词找到场景动词

            条件: 暂时通用, 目前发现的PI关系为advcl和xcomp
        """
        res = []

        # TODO 后面pattern成熟了替换掉
        if depRes[keyloc]["dep"] == "obl":
            return res

        fvloc = self._parseFinVerb(keyloc, depRes)
        if fvloc == -1:
            return res

        deplocs = self._parseDepWord(fvloc, depRes)
        if len(deplocs) == 0:
            return res
            
        for deploc in deplocs:
            finloc = self._findPhraseEnd(deploc, depRes)
            phrase = self._getPhrase(deploc, finloc, depRes)
            # PI, scene, findep, finverb, pattern
            res.append([depRes[keyloc]["dependent"], phrase, depRes[deploc]["dep"], depRes[fvloc]["dependent"], "pattern_2"])

        return res


    def parseDepRes(self, depRes):
        
        res = []

        depRes = self._formatDepRes(depRes)
        if depRes == None:
            return res
        
        keylocs = self._findWordLocs(PERM_KEYWORD_LIST, depRes)
        if len(keylocs) == 0:
            return res
        
        for keyloc in keylocs:
            
            tmpres = self._pattern1(keyloc, depRes)
            if len(tmpres) > 0:
                res.extend(tmpres)
                continue

            tmpres = self._pattern2(keyloc, depRes)
            res.extend(tmpres)

        return res

    def parseSentence(self, sentence):
        
        depRes = self.depParser.parse(sentence)
        self.depRes = depRes
        return self.parseDepRes(depRes)

    pass

if __name__ == "__main__":
    
    # ts = r"we may record your image through security cameras when you visit ASUS Royal Club repair stations and ASUS offices."
    # ts = r"Images recorded by cameras fitted to Sky's engineer vans."
    ts = r"Permission to access contact information is used when you search contacts in JVSTUDIOS search bar."
    # ts = r"The app needs access to the camera to fulfill recording videos."

    senParser = SentenceParser()
    res = senParser.parseSentence(ts)
    
    print(senParser.depParser.prettyRes(senParser.depRes))

    for e in res:
        print(e)

    # for g, d, dt in res.triples():
    #     print("%s(%s, %s) (%s, %s)" % (addSpaces(d), g[0], g[1], dt[0], dt[1]))

    # with open(r"./sentences.json", 'r', encoding="utf-8") as f:
    #     allSens = json.load(f)

    # for section in allSens:
    #     for item in section:
    #         sentence = item["sentences"]
    #         scene = item["view"]
    #         pi = item["privacy"]

    #         parseRes = depParser.parse(sentence)
    #         dep = []
    #         for g, d, dt in parseRes.triples():
    #             dep.append("%s(%s, %s) (%s, %s)" % (addSpaces(d), g[0], g[1], dt[0], dt[1]))
    
    #         resDict.append({
    #             "scene": scene,
    #             "PI": pi,
    #             "sentence": sentence,
    #             "dep": dep
    #         })
    
    # with open(r"./dep_sentence.json", 'w', encoding="utf-8") as f:
    #     json.dump(resDict, f, indent=4)