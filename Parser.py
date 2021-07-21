from json import decoder
import nltk
import json
from nltk.corpus.reader.wordnet import VERB
from nltk.tokenize import WordPunctTokenizer
from nltk.parse import corenlp
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from requests.models import parse_header_links

PERM_KEYWORD_LIST = ["contact", "address book",
                     "camera", "microphone", "record_audio",
                     "location", "longitude", "latitude", "GPS",
                     "SMS", "phone"]

PATTERN_1_DEP_LIST = ["obl", "appos"]
PATTERN_2_DEP_LIST = ["advcl", "xcomp", "obl", "obj", "nsubj", "conj"]
PATTERN_3_DEP_LIST = ["nmod", "obl"]
# TODO pattern_4
PATTERN_4_DEP_LIST = ["ccomp"]

FIFLTER_PATTERN = ["PRP","PRP$"]
# FIFLTER_PRP_WORDS = ["we","you","he","she","they","it"]
# FIFLTER_PRPS_WORDS = ["our","your","his","her","their","its"]

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
       目前将一个类型算作组合:
       1. compound(and its subtype)
    """
    if relation.startswith("compound"):
        return True
    
    # if relation == "amod":
    #     return True

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
            2. 如果 compound + 名词 + 连词 + 名词 的格式 会转化成 compound + 名词 + 连词 + compound + 名词
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
                # and 情况下 compound 补充：本身为名词且前面为名词有compound
                for index in range(gloc,len(depRes)):
                    if depRes[index]["dep"] == "conj" and depRes[index]["govloc"] == gloc and depRes[index]["pos"] == depRes[gloc]["pos"] and getPos(depRes[gloc]["pos"]) == wordnet.NOUN:
                        depRes[index]["dependent"] = "%s %s" % (dependent, depRes[index]["dependent"])
        
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

    def _parseGovFinVerb(self, nloc, depRes):
        """
            input_verb --> fin_verb
        """
        
        fvlocs = []
        govloc = depRes[nloc]["govloc"]
        govpos = depRes[govloc]["pos"]
        dep = depRes[govloc]["dep"]
        # dep == "obj" case: "The app needs access to the camera to fulfill recording videos."
        while getPos(govpos) == wordnet.VERB or dep == "obj":
            nloc = govloc
            govloc = depRes[nloc]["govloc"]
            govpos = depRes[govloc]["pos"]
            dep = depRes[govloc]["dep"]
            
            if getPos(depRes[nloc]["pos"]) == wordnet.VERB or \
               getPos(depRes[nloc]["pos"]) == wordnet.NOUN:
               fvlocs.append(nloc)
                
        return fvlocs

    def _parseDepWord(self, wloc, depRes):
        
        dlocs = []

        for idx, item in enumerate(depRes):
            govloc = item["govloc"]
            if govloc == wloc:
                dlocs.append(idx)

        return dlocs

    def _findConjWord(self, wloc, depRes):
        """
            找全部conj关系的并列词, 包含两种情况:
            1. 一是作为起始词, 在governor关系中找
            2. 二是作为后续词, 在自己的关系中找到起始词, 再进行第一步

            ! 暂不考虑递归查找
        """
        conjlocs = []

        # 找起始词
        start = wloc
        if depRes[wloc]["dep"] == "conj":
            start = depRes[wloc]["govloc"]
            conjlocs.append(start)
        
        # 从起始词找后续关系
        for idx, item in enumerate(depRes):

            govloc = item["govloc"]
            dep = item["dep"]

            if govloc != start or dep != "conj" or idx == wloc:
                continue
            
            conjlocs.append(idx)

        return conjlocs

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
        
        # 补全动词后面的名词   RP 指类似 give up 这种的词  出现的很少暂不考虑 · 
        while getPos(depRes[finloc]["pos"]) == wordnet.VERB:
            newloc = finloc
            distEnd = 1
            for dist in range(1, len(depRes)):

                left = newloc - dist
                right = newloc + dist

                if left >= 0:
                    govloc = depRes[left]["govloc"]
                    if govloc == newloc and not isInvalidPos(depRes[left]["pos"]):
                        finloc = left
                
                if right < len(depRes):
                    govloc = depRes[right]["govloc"]
                    if govloc == newloc and not isInvalidPos(depRes[right]["pos"]):
                        finloc = right
                
                if finloc != newloc:
                    break
                if dist == len(depRes) - 1:
                    distEnd = dist
            if distEnd == len(depRes) - 1:
                break

        return finloc

    # 处理连词问题
    def _getWholePhrase(self, start, end, depRes):
        if start > end:
            start, end = end, start

        # if getPos(depRes[finloc]["pos"]) == wordnet.VERB and depRes[finloc]["dep"] == "amod":
        #     govloc = depRes[finloc]["govloc"]
        #     if not isInvalidPos(depRes[govloc]["pos"]):
        #         finloc = govloc

        # 寻找目前词组内是否含有连词  一层连词
        last = end
        for dist in range(start, end + 1):
            for idx in range(dist, len(depRes)):
                govloc = depRes[idx]["govloc"]
                dep = depRes[idx]["dep"]
                if govloc == dist and dep == "conj" and idx > last:
                    last = idx
                if depRes[idx]["dependent"]  == "/" and idx > last:
                    last = idx + 1
        finloc = last
        
        # 有一个错误例子的匹配  or to enter search term  到enter结束后 补全动词
        while getPos(depRes[finloc]["pos"]) == wordnet.VERB :
            newloc = finloc
            distEnd = 1
            for dist in range(1, len(depRes)):

                left = newloc - dist
                right = newloc + dist

                if left >= 0:
                    govloc = depRes[left]["govloc"]
                    if govloc == newloc and not isInvalidPos(depRes[left]["pos"]):
                        distEnd = len(depRes) - 1
                        break
                
                if right < len(depRes):
                    govloc = depRes[right]["govloc"]
                    if govloc == newloc and not isInvalidPos(depRes[right]["pos"]):
                        finloc = right
                
                if finloc != newloc:
                    break
                if dist == len(depRes) - 1:
                    distEnd = dist
            if distEnd == len(depRes) - 1:
                break
        
        # if getPos(depRes[finloc]["pos"]) == wordnet.VERB and getPos(depRes[depRes[finloc]["govloc"]]["pos"]) == wordnet.VERB and depRes[finloc]["dep"] == "conj":
        #     conjsgovloc = depRes[finloc]["govloc"]
        #     newloc = finloc
        #     for dist in range(finloc + 1, len(depRes)):
        #         right = dist
                
        #         if right < len(depRes):
        #             govloc = depRes[right]["govloc"]
        #             if govloc == conjsgovloc and not isInvalidPos(depRes[right]["pos"]):
        #                 finloc = right
                
        #         if finloc != newloc:
        #             break

        # 补全之后第二层并列  第二层连词
        conjEnd = finloc
        for dist in range(last, finloc + 1):
            for idx in range(dist, len(depRes)):
                govloc = depRes[idx]["govloc"]
                dep = depRes[idx]["dep"]
                if govloc == dist and dep == "conj" and idx > conjEnd:
                    conjEnd = idx
        finloc = conjEnd

        # 介词抓取
        if finloc + 1 < len(depRes):
            # 只考虑via 和 during
            if depRes[finloc + 1]["dependent"] == "via" or depRes[finloc + 1]["dependent"] == "during":
                govloc = depRes[finloc + 1]["govloc"]
                if finloc < govloc:
                    finloc = govloc

        end = finloc

        return start,end

    def _isConj(self, idx,depRes):
        isConj = 0
        if (depRes[idx]["pos"] == "," and depRes[idx]["dependent"] != "/"):        
            for dist in range(idx, len(depRes)):
                if depRes[dist]["dep"] == "cc":
                    break
                elif depRes[dist]['dep'] == "conj":
                    isConj = dist
                    break       
        elif depRes[idx]["dep"] == "cc":
            isConj = 1

        return isConj
            

    def _conjsType(self,start,end,depRes):
        types = {}
        indexs = set()
        for idx in range(start, end + 1):
            isConj = self._isConj(idx,depRes)
            if isConj !=0:
                if isConj ==1:
                    govloc = depRes[idx]['govloc']
                    # 找到第一个连词
                    conjsloc = depRes[govloc]['govloc']
                else:
                    conjsloc = depRes[idx]['govloc']
                    govloc = isConj
                # 词性相同才考虑补全
                if getPos(depRes[govloc]['pos']) == getPos(depRes[conjsloc]['pos']):
                    # 动词并列考虑后宾语补全
                    if getPos(depRes[govloc]['pos']) == wordnet.VERB:
                        for idx2 in range(idx + 1,len(depRes)):
                            # 找宾语
                            if getPos(depRes[idx2]['pos']) == wordnet.NOUN and depRes[idx2]['govloc'] == conjsloc:
                                types[conjsloc] = idx2
                                types[govloc] = idx2
                                indexs.add(conjsloc)
                                indexs.add(govloc)
                                break
                    # 名词并列
                    if getPos(depRes[govloc]['pos']) == wordnet.NOUN:
                            types[govloc] = 1
                            indexs.add(govloc)
        return types,indexs 


    def _getPhrase(self, start, end, depRes):
        
        phrase = []

        if start > end:
            start, end = end, start

        types,indexs = self._conjsType(start,end,depRes)

        # 不存在并列
        if len(indexs) == 0:
            for idx in range(start, end + 1):
                if (depRes[idx]["pos"] == "," and depRes[idx]["dependent"] != "/") or depRes[idx]["dep"] == "cc":
                    phrase.append("@#$%^&")
                else:
                    phrase.append(depRes[idx]["dependent"])
            return ' '.join(phrase)

        # 存在并列
        for idx in range(start, end + 1):
            if (depRes[idx]["pos"] == "," and depRes[idx]["dependent"] != "/") or depRes[idx]["dep"] == "cc":
                phrase.append("@#$%^&")
            else:
                # 名词的情况，本身是名词 补全两种：前面的compond（在一开始时完成） 前面的动词或者介词词组
                if idx in indexs and types[idx] == 1:
                    conjs = depRes[idx]["govloc"]
                    # obj 为前面有动词的情况 或者 介词词组 nmod 的情况
                    if (depRes[conjs]["dep"] == "obj" and getPos(depRes[depRes[conjs]["govloc"]]["pos"]) == wordnet.VERB) or depRes[conjs]["dep"] == "nmod": 
                        for idx2 in range(depRes[conjs]["govloc"], conjs):
                            # 排除：microphone permission : for recording …… 这种前面介词词组的误差  这里介词只允许是of
                            if depRes[idx2]["pos"] == "IN" and depRes[idx2]["dependent"] != "of" or depRes[idx2]["dep"] == "punct":
                                phrase = phrase[:(depRes[conjs]["govloc"] - idx2)]
                                break
                            else:
                                phrase.append(depRes[idx2]["dependent"])
                    else:
                        # 还有一种情况是前面是动词 但是amod 格式 组合  有两种解决方案 一种和compound一样在前面合并 一种在这里找到
                        isAmod = 0
                        for idx2 in range(depRes[conjs]["govloc"], conjs):
                            if depRes[idx2]["dep"] == "amod" and depRes[idx2]["govloc"] == conjs and conjs - idx2 <=2:
                                isAmod = 1
                            if isAmod == 1:
                                phrase.append(depRes[idx2]["dependent"])
                    phrase.append(depRes[idx]["dependent"])
                # 动词补全
                elif idx in indexs and types[idx] != 1:
                    objs = types[idx] 
                    lastVerb = idx
                    for key,values in types.items():
                        if values == objs and lastVerb < key:
                            lastVerb = key
                    phrase.append(depRes[idx]["dependent"])
                    for idx2 in range(lastVerb + 1, objs + 1):
                        phrase.append(depRes[idx2]["dependent"])
                else:
                    phrase.append(depRes[idx]["dependent"])

        # for idx in range(start, end + 1):
        #     if (depRes[idx]["pos"] == "," and depRes[idx]["dependent"] != "/") or depRes[idx]["dep"] == "cc":
        #         phrase.append("@#$%^&")
        #     else:
        #         # 连词 分开
        #         if depRes[idx]["dep"] == "conj":
        #             # 找到第一个连词
        #             conjs = depRes[idx]["govloc"]
        #             # 补全： 本身是名词 补全两种：前面的compond（在一开始时完成） 前面的动词或者介词词组
        #             # 连词后面补全暂时不考虑
        #             if getPos(depRes[conjs]["pos"]) == wordnet.NOUN and getPos(depRes[idx]["pos"]) == wordnet.NOUN:
        #                 # obj 为前面有动词的情况 或者 介词词组 nmod 的情况
        #                 if (depRes[conjs]["dep"] == "obj" and getPos(depRes[depRes[conjs]["govloc"]]["pos"]) == wordnet.VERB) or depRes[conjs]["dep"] == "nmod": 
        #                     for idx2 in range(depRes[conjs]["govloc"], conjs):
        #                         # 排除：microphone permission : for recording …… 这种前面介词词组的误差  这里介词只允许是of
        #                         if depRes[idx2]["pos"] == "IN" and depRes[idx2]["dependent"] != "of" or depRes[idx2]["dep"] == "punct":
        #                             phrase = phrase[:(depRes[conjs]["govloc"] - idx2)]
        #                             break
        #                         else:
        #                             phrase.append(depRes[idx2]["dependent"])
        #                 else:
        #                     # 还有一种情况是前面是动词 但是amod 格式 组合  有两种解决方案 一种和compound一样在前面合并 一种在这里找到
        #                     isAmod = 0
        #                     for idx2 in range(depRes[conjs]["govloc"], conjs):
        #                         if depRes[idx2]["dep"] == "amod" and depRes[idx2]["govloc"] == conjs and conjs - idx2 <=2:
        #                             isAmod = 1
        #                         if isAmod == 1:
        #                             phrase.append(depRes[idx2]["dependent"])
                    # else:
                    #     if getPos(depRes[conjs]["pos"]) == wordnet.VERB and getPos(depRes[idx]["pos"]) == wordnet.VERB:
                    #         for dist in range(end + 1, len(depRes)):
                    #             right = dist    
                    #             if right < len(depRes):
                    #                 govloc = depRes[right]["govloc"]
                    #                 if govloc == conjs and not isInvalidPos(depRes[right]["pos"]):
                    #                     finloc = right
                    #                     conjEnd = finloc
                    #                     for dist2 in range(finloc, finloc + 1):
                    #                         for idx in range(dist2, len(depRes)):
                    #                             govloc = depRes[idx]["govloc"]
                    #                             dep = depRes[idx]["dep"]
                    #                             if govloc == right and dep == "conj" and idx > conjEnd:
                    #                                 conjEnd = idx
                    #                     finloc = conjEnd
                    #                     for idx2 in range(right, conjEnd):
                    #                         phrase.append(depRes[idx2]["dependent"])
                    #                     break
        
        return ' '.join(phrase)

    def _pattern1(self, keyloc, depRes):
        """
            case(obl): Images recorded by cameras fitted to Sky's engineer vans.
            4 (cameras, NNS) obl [2](recorded, VBN)

            case(appos): Recording Call, Microphone
            3 (microphone, NNP) appos [1](recording call, NNP) 

            利用PI直接找到场景动词
            
            条件: PI的依赖关系为以上两种
        """

        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)
        
        for conjloc in conjlocs:
            dep = depRes[conjloc]["dep"].split(':')[0]
            if dep not in PATTERN_1_DEP_LIST:
                continue

            deploc = depRes[conjloc]["govloc"]
            finloc = self._findPhraseEnd(deploc, depRes)
            deploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
            phrase = self._getPhrase(deploc, finloc, depRes)
            phrases = phrase.split('@#$%^&')
            phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
            res.append([depRes[keyloc]["dependent"], phrases, 
                        depRes[conjloc]["dep"], depRes[conjloc]["dependent"], 
                        "pattern_1", depRes[conjloc]["dependent"]])
            # res.append([depRes[keyloc]["dependent"], phrase, 
            #             depRes[conjloc]["dep"], depRes[conjloc]["dependent"], 
            #             "pattern_1", depRes[conjloc]["dependent"]])

        return res
    
    def _pattern2(self, keyloc, depRes):
        """
            case(advcl): Permission to access contact information is used when you search contacts in JVSTUDIOS search bar.
            4  (contact, NN) compound [5](information, NN)
            10 (search, VBP) advcl    [7](used, VBN)

            case(xcomp): The app needs access to the camera to fulfill recording videos.
            7 (camera, NN)   nmod  [4](access, NN)
            9 (fulfill, VB)  xcomp [3](needs, VBZ)

            case(obl): The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms.
            4 (microphone, NNS) nsubj [5](enable, VBP)
            8 (navigation, NN)  obl [5](enable, VBP)

            case(obj): Microphone, To enable voice command related actions.
            1 (microphone, NN)                   nsubj  [4](enable, VB)
            5 (voice command relate action, NNS) obj    [4](enable, VB)

            case(nsubj:pass): Voice control features will only be enabled if you affirmatively activate voice controls by giving us access to the microphone on your device.
            1  (voice control feature, NNS) nsubj:pass  [5](enable, VBN)
            17 (microphone, NN)             nmod        [14](access, NN)

            利用PI找到依赖动词, 利用依赖动词找到场景动词

            条件: 暂时通用, 目前发现的PI关系为以上6种
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:
            fvlocs = self._parseGovFinVerb(conjloc, depRes)
            if len(fvlocs) == 0:
                continue

            for fvloc in fvlocs:
                deplocs = self._parseDepWord(fvloc, depRes)
                if len(deplocs) == 0:
                    continue
                    
                for deploc in deplocs:

                    if deploc in fvlocs or deploc in conjlocs or depRes[deploc]["pos"] in FIFLTER_PATTERN:
                        continue
                    
                    dep = depRes[deploc]["dep"].split(':')[0]
                    if dep not in PATTERN_2_DEP_LIST:
                        continue
                    # if dep == "nsubj" and depRes[deploc]["pos"] in FIFLTER_PATTERN:
                    #     continue

                    finloc = self._findPhraseEnd(deploc, depRes)
                    deploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                    phrase = self._getPhrase(deploc, finloc, depRes)
                    # PI, scene, findep, finverb, patterns
                    phrases = phrase.split('@#$%^&')
                    phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                    res.append([depRes[keyloc]["dependent"], phrases, 
                                    depRes[deploc]["dep"], depRes[fvloc]["dependent"], 
                                    "pattern_2", depRes[conjloc]["dependent"]])

        return res

    def _pattern3(self, keyloc, depRes):
        """
            case(nmod): When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Microphone, for recording voices in the videos.
            32 (record voice, NNS) nmod [29](microphone, NNP)

            利用PI做governor来直接找到对应的场景

            条件: nmod关系
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:        
            deplocs = self._parseDepWord(conjloc, depRes)
            if len(deplocs) == 0:
                continue

            for deploc in deplocs:
                if depRes[deploc]["pos"] in FIFLTER_PATTERN:
                    continue

                dep = depRes[deploc]["dep"].split(':')[0]
                if not dep in PATTERN_3_DEP_LIST:
                    continue
                
                #  为何提取要拆分成两部分
                #  When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Cameras, for shooting photos and taking videos.
                #  camera 定位到photo 则 要从shoot到videos 需要两边都修改，所以要能返回两个值
                finloc = self._findPhraseEnd(deploc, depRes)
                deploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                phrase = self._getPhrase(deploc, finloc, depRes)

                # res.append([depRes[keyloc]["dependent"], phrase, 
                #             depRes[deploc]["dep"], depRes[conjloc]["dependent"], 
                #             "pattern_3", depRes[conjloc]["dependent"]])
                phrases = phrase.split('@#$%^&')
                phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                
                res.append([depRes[keyloc]["dependent"], phrases, 
                            depRes[deploc]["dep"], depRes[conjloc]["dependent"], 
                            "pattern_3", depRes[conjloc]["dependent"]])

        return res

    # 过滤  暂时过滤
    def filter(self,tmpres):
        finalpres = []
        for phrases in tmpres:
            if len(''.join(phrases[1])) > 100 or len(''.join(phrases[1])) < 3:
                continue
            # if phrases[2] == "nmod:poss":  本来基本上nmod:poss都是所有格your，但是发现了一个例外The Product's meeting functionality also enables you to be seen by other participants through your built-in device camera.
            #     continue
            finalpres.append(phrases)
        return finalpres

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
            res.extend(self.filter(tmpres))

            tmpres = self._pattern2(keyloc, depRes)
            res.extend(self.filter(tmpres))

            tmpres = self._pattern3(keyloc, depRes)
            res.extend(self.filter(tmpres))

        return res

    def parseSentence(self, sentence):
        
        depRes = self.depParser.parse(sentence)
        self.depRes = depRes
        return self.parseDepRes(depRes)

    pass

if __name__ == "__main__":

    senParser = SentenceParser()    
    # ts = r"If you wish to invite your friends and contacts to use the Services, we will give you the option of either entering in their contact information manually."
    # ts = r"The app accesses the contacts on the phone in order to display the contacts and so the app can record or ignore calls from specific contacts."
    # ts = r"We display all the phone calls in the phone list in the form of lists, and you can dial the phone directly in the message center. In order to make you use this function normally, we need to get call record information to implement the display and edit management of the call record list, including the telephone number, the way to call (the incoming calls, the unanswered calls, the dialed and rejected calls), and the time of the call. Meanwhile, in order to help you quickly select contacts in dialing, we need to read your address book contact information."
    # ts = r"The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms. Learn more about voice data collection."
    # ts = r"If granted permission by a user, My Home Screen Apps uses access to a device's microphone to facilitate voice-enabled search queries and may access your deviceâs camera, photos, media, and other files."
    # ts = r"We may also collect contact information for other individuals when you use the sharing and referral tools available within some of our Services to forward content or offers to your friends and associates."
    # ts = r"We collect the following permissions and corresponding information following the criterion of minimal data collection with your consent: Microphone permissions: for video shooting and editing."
    # ts = r"Take pictures and videos absurd Labs need this permission to use your phone's camera to take pictures and videos."
    # ts = r"This permission allows JVSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash."


    # ts = r"The microphone also enables voice commands for control of the console, game, or app, or to enter search terms."
    # ts = r"Using your microphone for making note via voice."
    # ts =r"When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Cameras, for shooting photos and taking videos."
    # ts =r"We access the microphone on your device (with your permission) to record audio messages and deliver sound during video calls."
    # ts = r"With your prior consent we will be allowed to use the microphone for songs immediate identification and lyrics synchronization."
    # ts =r"TomTom navigation products and services need location data and other information to work correctly."
    # ts = r"Some of our Apps offer you the option to talk to the virtual character of such Apps. All of our Apps that offer such feature will always ask you for your permission to access the microphone on your device in advance. If you decide to give us the permission, the virtual character will be able to repeat what you say to him. Please note that our Apps do not have a function to record audio, so what you say to the virtual character is not stored on our servers, it is only used by the App to repeat to you in real time. If you decide not to give us the permission to access the microphone on your device, the virtual character will not be able to repeat after you."
    # ts = r"The Product's meeting functionality also enables you to be seen by other participants through your built-in device camera."
    # ts = r"Cameras and photos: in order to be able to send and upload your photos from your camera, you must give us permission to access them."

    # ts = r"Images recorded by cameras fitted to Sky's engineer vans."
    # ts = r"The app needs access to the camera to fulfill recording videos."
    # ts = r"If granted permission by a user, we use access to a phone's microphone to facilitate voice enabled search queries. All interaction and access to the microphone is user initiated and voice queries are not shared with third party apps or providers."
    # ts = r"Some features like searching a contact from the search bar require access to your Address book."
    # ts = r"including your public profile, the lists you create, and photos, videos and voice recordings as accessed with your prior consent through your device's camera and microphone sensor."
    # ts = r"Voice control features will only be enabled if you affirmatively activate voice controls by giving us access to the microphone on your device."
    # ts = r"The app needs access to the camera to fulfill recording videos."
    # ts = r"Some features like searching a contact from the search bar require access to your Address book."
    # ts = r"When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Microphone, for recording voices in the videos."
    # ts = r"We collect the following permissions and corresponding information following the criterion of minimal data collection with your consent: â· Microphone permissions: for video shooting and editingï¼"
    # ts = r"Microphone, for recording voices in the videos."
    # ts = r"Recording Call, Microphone"
    # ts = r"Microphone; for detecting your voice and command,"
    # ts = r"Used for accessing the camera or capturing images and video from the device."
    # ts = r"The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms."
    # ts = r"HoloLens also processes and collects data related to the HoloLens experience and device, which include cameras, microphones, and infrared sensors that enable motions and voice to navigate."
    # ts = r"If you wish to invite your friends and contacts to use the Services, we will give you the option of either entering in their contact information manually."
    # ts = r"As Offline Map Navigation app is a GPS based navigation application which uses your location while using the app or all the time."
    # ts = r"including your public profile, the lists you create, and photos, videos and voice recordings as accessed with your prior consent through your device's camera and microphone sensor"
    # ts = r"Permission to access contact information is used when you search contacts in JVSTUDIOS search bar."
    # ts = r"Used to give sites ability to ask users to utilize microphone and used to provide voice search feature."
    # ts = r"Some features like searching a contact from the search bar require access to your Address book."
    # ts = r"Camera; for taking selfies and pictures using voice."

    res = senParser.parseSentence(ts)  
    
    print(senParser.depParser.prettyRes(senParser.depRes))
    for e in res:
        print(e)

    # ds = r"As Offline Map Navigation app is a GPS based navigation application which uses your location while using the app or all the time."
    # ds = r"the sharing and referral tools"
    # depParser = DepParser()
    
    # res = depParser.parse(ds)
    # print(depParser.prettyRes(res))

    # with open(r"./sentences.json", 'r', encoding="utf-8") as f:
    # with open(r"../test.json", 'r', encoding="utf-8") as f:
    #     allSens = json.load(f)

    # with open(r"./dep_sentence_5.txt", 'w', encoding="utf-8") as f:
    #     index = 0
    #     resDict = []
    #     for item in allSens:
    #         # for item in section:
    #         sentence = item["sentence"]
    #         scene = item["scene"]
    #         pi = item["PI"]

    #         res = senParser.parseSentence(sentence)
    #         parseRes = senParser.depParser.prettyRes(senParser.depRes)

    #         resDict.append({
    #             "scene": scene,
    #             "PI": pi,
    #             "sentence": sentence,
    #             "parse": res,
    #             "dep": parseRes
    #         })

    #         # write
    #         f.write("%s. %s\n" % (str(index), sentence))
    #         f.write("\n")
    #         f.write("%s -> %s\n" % (str(pi), str(scene)))
    #         f.write("\n")
    #         for e in res:
    #                 # f.write("%s -> %s, %s, %s, %s" % (e[0], e[1], e[2], e[3], e[4]))
    #             try:
    #                 f.write("%s -> %s,\t%s,\t%s,\t%s\t%s\n" % tuple(e))
    #             except:
    #                 f.write("error:"+ str(e))
    #         f.write("\n")
    #         f.write(parseRes)
    #         f.write("\n=====================\n\n")
    #         index += 1

    # with open(r"./dep_sentence.json", 'w', encoding="utf-8") as f:
    #     json.dump(resDict, f, indent=4)
