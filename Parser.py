import nltk
from nltk.corpus.reader.wordnet import VERB
from nltk.parse import corenlp
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

PERM_KEYWORD_LIST = ["contact", "address book",
                     "camera", "microphone", "record_audio",
                     "location", "longitude", "latitude", "GPS",
                     "SMS", "phone"]

PATTERN_2_DEP_LIST = ["xcomp", "nsubj:pass", "advcl", "nsubj"]

FIFLTER_PATTERN = ["PRP","PRP$"]

IN_WORDS_LIST = ['for','during','via']

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

    def prettyResList(self, usrinput):

        if type(usrinput) == str:
            usrinput = self.parse(usrinput)
    
        resstrlist = []
        for idx, item in enumerate(usrinput):
            dependent = item["dependent"]
            pos = item["pos"]
            governorloc = item["govloc"]
            dep = item["dep"]
            governor = usrinput[governorloc]["dependent"]
            governorpos = usrinput[governorloc]["pos"]

            resstrlist.append("%d   %s%s%s" % (idx, 
                     addSpaces("(%s, %s)" % (dependent, pos), 24),
                    addSpaces(dep, 16),
                    addSpaces("[%d](%s, %s)" % (governorloc, governor, governorpos), 28)))

        return resstrlist
    pass

def addSpaces(s, length):
    
    if len(s) < length:
        s += " " * (length - len(s))

    return s

def isCompound(relation):
    """
       组合关系, 相邻两个词可以组合为短语.
       目前将一个类型算作组合:
       1. compound
    """
    if relation == "compound":
        return True
    
    # if relation == "amod":
    #     return True

    return False

def isFixed(relation):
    """
       组合关系, 相邻两个词可以组合为短语, 和compound方向相反
       目前将一个类型算作组合:
       2. fixed 暂时全视作介词词性
    """
    if relation == "fixed":
        return True
    
    return False

def isInvalidPos(pos):
    # TODO 暂时只关注四个词性，后面有问题再改
    if getPos(pos) == "":
        return True

    return False

def hashTuple(res):
    # 计算分析二元组的hash值
    text = res[0] + str(res[1]) + str(res[3])
    # text = res[0] + str(res[1])
    return hash(text)

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
            1. 合并全部**相邻**compound关系, 包括sub-compund
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

            if isCompound(dep):
                
                siblinggloc = findTrueGov(idx + 1)
                if siblinggloc == gloc:
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
            
            locMap[idx] = delta
            if idx != gloc:
                # 删除节点
                del(depRes[idx - delta])
                delta += 1    
        
        for idx, item in enumerate(depRes):
            depRes[idx]["govloc"] -= locMap[depRes[idx]["govloc"]]
        
        return depRes

    def _formatDepRes2(self, depRes):
        """
            预处理依存关系结果:
            1. 合并全部**相邻**/fixed/flat关系, 包括sub-compund
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

            if isFixed(dep):
                siblinggloc = findTrueGov(idx - 1)
                if siblinggloc == gloc:
                    locMap[idx] = findTrueGov(depRes[gloc]["govloc"])
                    depRes[gloc]["dependent"] = "%s %s" % (depRes[gloc]["dependent"], dependent)
                    # 暂时全认为是介词
                    depRes[gloc]["pos"] = "IN"

            elif depRes[idx]['dep'] == "compound:prt" and getPos(depRes[gloc]["pos"]) == wordnet.VERB:
                siblinggloc = findTrueGov(idx - 1)
                if siblinggloc == gloc:
                    locMap[idx] = findTrueGov(depRes[gloc]["govloc"])
                    # 更新gov
                    depRes[gloc]["dependent"] = "%s %s" % (depRes[gloc]["dependent"], dependent)

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
            pos = getPos(item["pos"])
            # TODO 后面可以换相似度之类的方法, 现在就只是匹配一哈
            for key in keywords:
                if pos != wordnet.NOUN:
                    continue
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

    def _findTargetDepWord(self, wloc, depRes, tardeps):

        dlocs = []

        for idx, item in enumerate(depRes):
            govloc = item["govloc"]
            dep = item["dep"].split(':')[0]            
            if govloc == wloc and dep in tardeps:
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
        if depRes[wloc]["dep"] == "conj" or depRes[wloc]["dep"] == "csubj:pass":
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

    def _findDirectVerb(self, wloc, depRes):
        """
            匹配PI的直接依赖动词, 比如use camera
        """

        # 如果存在动宾词组, 补充动词, *动词在名词前面*
        govloc = depRes[wloc]["govloc"]
        # access 有时会被当成名词, 
        if getPos(depRes[govloc]["pos"]) == wordnet.VERB or depRes[govloc]["dependent"] == "access":
            return govloc

        return None

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

        
        # 介词抓取  
        if end + 1 < len(depRes):
            # 所有的介词都提取了
            # if depRes[finloc + 1]["pos"] == "IN" and depRes[finloc + 1]["dep"] == "case":
            # 只考虑for via 和 during
            if depRes[end + 1]["dependent"] in IN_WORDS_LIST and depRes[end + 1]["pos"] == "IN" and depRes[end + 1]["dep"] == "case":
                govloc = depRes[end + 1]["govloc"]
                if end < govloc:
                    end = govloc

        # 是否自身为conj
        for dist in range(start, end + 1):
            govloc = depRes[dist]["govloc"]
            dep = depRes[dist]["dep"]
            if govloc != dist and dep == "conj" and getPos(depRes[govloc]['pos']) == getPos(depRes[dist]['pos']) and govloc < start:
                start = govloc

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

        if getPos(depRes[finloc]["pos"]) == wordnet.VERB and depRes[finloc]['dep'] == "amod" and  getPos(depRes[depRes[finloc]["govloc"]]["pos"]) == wordnet.NOUN and depRes[finloc]["govloc"] > finloc:
            finloc = depRes[finloc]["govloc"]
     
        
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
            if depRes[finloc + 1]["dependent"] in IN_WORDS_LIST and depRes[finloc + 1]["pos"] == "IN" and depRes[finloc + 1]["dep"] == "case":
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
        
        return ' '.join(phrase)
    
    def _findClosedVerb(self, wloc, depRes, limitDeps):
        # 找到与wloc最近的动词
        while wloc != 0:
            dep = depRes[wloc]["dep"]
            govloc = depRes[wloc]["govloc"]

            if getPos(depRes[govloc]["pos"]) != wordnet.VERB:
                wloc = govloc
            else:
                if dep in limitDeps:
                    return govloc
                elif depRes[govloc]["dep"] in ["acl"]:
                    wloc = depRes[govloc]["govloc"]
                    continue
                break

        return None

    def isRepeat(self, start, end, conjlocs):
        
        delta = end - start
        for conj in conjlocs:
            if 0 <= conj - start <= delta:
                return True
        
        return False

    def _pattern0(self, keyloc, depRes):
        """
            pattern: (PI) [verb]{enable} [verb]{doing} (SCENE)
            case: The location infomation can enable navigation.
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:
            fvloc = self._findClosedVerb(conjloc, depRes, ["nsubj"])

            deplocs = self._parseDepWord(fvloc, depRes)
            if len(deplocs) <= 0:
                continue

            for deploc in deplocs:
                if depRes[deploc]["dep"] != "obj":
                    continue

                # 判断场景目标词的词性
                deppos = getPos(depRes[deploc]["pos"])
                if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                    continue

                finloc = self._findPhraseEnd(deploc, depRes)
                deploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                phrase = self._getPhrase(deploc, finloc, depRes)
                phrases = phrase.split('@#$%^&')
                phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                res.append([depRes[keyloc]["dependent"], phrases, 
                            depRes[conjloc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc),
                            "pattern_0(%s)" % depRes[conjloc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])
                    
        return res

    def _pattern1(self, keyloc, depRes):
        """
            pattern: (PI) [verb]{enable} [verb]{doing} (SCENE)
            case: The location infomation can enable optimizing navigation.
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:
            fvloc = self._findClosedVerb(conjloc, depRes, ["nsubj"])

            depvlocs = []
            depvlocs.extend(self._findTargetDepWord(fvloc, depRes, ["dep", "nmod"]))

            for depvloc in depvlocs:
                
                deplocs = self._parseDepWord(depvloc, depRes)
                if len(deplocs) <= 0:
                    continue
                for deploc in deplocs:
                    
                    if depRes[deploc]["dep"] not in ["obj"]:
                        continue
                    
                    # 判断场景目标词的词性
                    deppos = getPos(depRes[deploc]["pos"])
                    if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                        continue

                    finloc = self._findPhraseEnd(deploc, depRes)
                    deploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                    if self.isRepeat(deploc, finloc, [conjloc]):
                        continue

                    phrase = self._getPhrase(deploc, finloc, depRes)
                    phrases = phrase.split('@#$%^&')
                    phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                    res.append([depRes[keyloc]["dependent"], phrases, 
                                depRes[conjloc]["dep"], "%s[%d]" % (depRes[depvloc]["dependent"], depvloc),
                                "pattern_1(%s)" % depRes[conjloc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])
                    
        return res

    def _pattern2(self, keyloc, depRes):
        """
            pattern: (SCENE) [prep]{by} (PI)
            case: Image recorded by your camera.
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)
        
        for conjloc in conjlocs:

            dep = depRes[conjloc]["dep"].split(':')[0]
            if dep != "obl":
                continue

            deploc = depRes[conjloc]["govloc"]
            # 判断 govloc 和 deploc 之间是否存在介词
            hasPreposision = False
            for loc in range(deploc + 1, conjloc):
                if depRes[loc]["dep"] == "case" and depRes[loc]["pos"] == "IN":
                    hasPreposision = True
                    break
            if not hasPreposision:
                continue

            # 判断场景目标词的词性
            deppos = getPos(depRes[deploc]["pos"])
            if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                continue

            finloc = self._findPhraseEnd(deploc, depRes)
            deploc,finloc = self._getWholePhrase(deploc, finloc, depRes)
            if self.isRepeat(deploc, finloc, [conjloc]):
                continue

            phrase = self._getPhrase(deploc, finloc, depRes)
            phrases = phrase.split('@#$%^&')
            phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
            res.append([depRes[keyloc]["dependent"], phrases, 
                        depRes[conjloc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc),
                        "pattern_2(%s)" % depRes[conjloc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res
    
    def _pattern3(self, keyloc, depRes):
        """
            pattern: (PI) [prep]{for} (SCENE)
            case: Microphone, for recording voices in the videos.
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:
            
            deplocs = []
            deplocs.extend(self._findTargetDepWord(conjloc, depRes, ["dep", "nmod", "obl"]))

            for deploc in deplocs:

                if deploc in conjlocs:
                    continue

                # 判断 govloc 和 deploc 之间是否存在介词
                hasPreposision = False
                for loc in range(conjloc + 1, deploc):
                    if depRes[loc]["dep"] == "case" and depRes[loc]["pos"] == "IN":
                        hasPreposision = True
                        break
                if not hasPreposision:
                    continue

                # 判断场景目标词的词性
                deppos = getPos(depRes[deploc]["pos"])
                if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                    continue

                finloc = self._findPhraseEnd(deploc, depRes)
                tdeploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                if self.isRepeat(tdeploc, finloc, [conjloc]):
                    continue

                phrase = self._getPhrase(tdeploc, finloc, depRes)
                phrases = phrase.split('@#$%^&')
                phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                res.append([depRes[keyloc]["dependent"], phrases, 
                            depRes[deploc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc), 
                            "pattern_3(%s)" % depRes[deploc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res
    
    def _pattern4(self, keyloc, depRes):
        """
            pattern: [verb]{use} (PI) [prep]{for} (SCENE)
            case: Using your microphone for making note via voice.
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:

            vloc = self._findDirectVerb(conjloc, depRes)
            deplocs = []
            deplocs.extend(self._findTargetDepWord(vloc, depRes, ["obl", "nmod", "det"]))

            for deploc in deplocs:

                if deploc in conjlocs:
                    continue

                # 判断 govloc 和 deploc 之间是否存在介词
                hasPreposision = False
                for loc in range(conjloc + 1, deploc):
                    if depRes[loc]["dep"] == "case" and depRes[loc]["pos"] == "IN":
                        hasPreposision = True
                        break
                if not hasPreposision:
                    continue

                # 判断场景目标词的词性
                deppos = getPos(depRes[deploc]["pos"])
                if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                    continue

                finloc = self._findPhraseEnd(deploc, depRes)
                tdeploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                if self.isRepeat(tdeploc, finloc, [conjloc]):
                    continue

                phrase = self._getPhrase(tdeploc, finloc, depRes)
                phrases = phrase.split('@#$%^&')
                phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                res.append([depRes[keyloc]["dependent"], phrases, 
                            depRes[deploc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc), 
                            "pattern_4(%s)" % depRes[deploc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res

    def _pattern5(self, keyloc, depRes):
        """
           pattern: [verb]{use} (PI) [prep]{to} (SCENE)
           case: CAMERA is required to let the app take pictures.
        """

        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:
            fvloc = self._findClosedVerb(conjloc, depRes, ["obj", "nsubj:pass"])
            if not fvloc:
                continue

            fvlocs = self._findConjWord(fvloc, depRes)
            fvlocs.append(fvloc)

            for fvloc in fvlocs:
                deplocs = self._parseDepWord(fvloc, depRes)
                if len(deplocs) == 0:
                    continue
                        
                for deploc in deplocs:
                    
                    if deploc == keyloc:
                        continue

                    dep = depRes[deploc]["dep"]
                    if dep != "xcomp":
                        continue

                    # 判断场景目标词的词性
                    deppos = getPos(depRes[deploc]["pos"])
                    if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                        continue

                    # 判断 govloc 和 deploc 之间是否存在介词
                    hasPreposision = False
                    for loc in range(fvloc + 1, deploc):
                        if depRes[loc]["pos"] in ["IN", "TO"]:
                            hasPreposision = True
                            break
                    if not hasPreposision:
                        continue

                    finloc = self._findPhraseEnd(deploc, depRes)
                    deploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                    if self.isRepeat(deploc, finloc, [conjloc]):
                        continue

                    phrase = self._getPhrase(deploc, finloc, depRes)
                    phrases = phrase.split('@#$%^&')
                    phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                    res.append([depRes[keyloc]["dependent"], phrases, 
                                depRes[deploc]["dep"],"%s[%d]" % (depRes[fvloc]["dependent"], fvloc), 
                                "pattern_5(%s)" % depRes[deploc]["dep"], 
                                "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res

    def _pattern6(self, keyloc, depRes):
        """
           pattern: (SCENE) [verb]{need} (PI)
           case: TomTom navigation products and services need location data and other information to work correctly.
        """

        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:
            fvloc = self._findClosedVerb(conjloc, depRes, ["obj", "nsubj:pass"])
            if not fvloc:
                continue

            fvlocs = self._findConjWord(fvloc, depRes)
            fvlocs.append(fvloc)

            for fvloc in fvlocs:
                deplocs = self._parseDepWord(fvloc, depRes)
                if len(deplocs) == 0:
                    continue
                        
                for deploc in deplocs:
                    
                    if deploc == keyloc:
                        continue

                    dep = depRes[deploc]["dep"]
                    if dep != "nsubj":
                        continue

                    # 判断场景目标词的词性
                    deppos = getPos(depRes[deploc]["pos"])
                    if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                        continue

                    finloc = self._findPhraseEnd(deploc, depRes)
                    tdeploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                    if self.isRepeat(tdeploc, finloc, [conjloc]):
                        continue                    

                    phrase = self._getPhrase(tdeploc, finloc, depRes)
                    phrases = phrase.split('@#$%^&')
                    phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                    res.append([depRes[keyloc]["dependent"], phrases, 
                                depRes[deploc]["dep"],"%s[%d]" % (depRes[fvloc]["dependent"], fvloc), 
                                "pattern_6(%s)" % depRes[deploc]["dep"], 
                                "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res
        
    def _pattern7(self, keyloc, depRes):
        """
           pattern: [verb]{collect} (PI) [ADVMOD]{when} (SCENE)
           case: We may also collect contact information for other individuals when you use the sharing and referral tools available within some of our Services to forward content or offers to your friends and associates.
        """

        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)

        for conjloc in conjlocs:
            fvloc = self._findClosedVerb(conjloc, depRes, ["obj", "nsubj:pass"])
            if not fvloc:
                continue

            fvlocs = self._findConjWord(fvloc, depRes)
            fvlocs.append(fvloc)

            for fvloc in fvlocs:
                deplocs = self._parseDepWord(fvloc, depRes)
                if len(deplocs) == 0:
                    continue
                        
                for deploc in deplocs:
                    
                    if deploc == keyloc:
                        continue

                    dep = depRes[deploc]["dep"]
                    if dep != "advcl":
                        continue

                    # 判断场景目标词的词性
                    deppos = getPos(depRes[deploc]["pos"])
                    if deppos != wordnet.NOUN and deppos != wordnet.VERB:
                        continue

                    # 判断 govloc 和 deploc 之间是否存在介词
                    hasPreposision = False
                    for loc in range(0, deploc):
                        if depRes[loc]["pos"] in ["IN", "TO", "WRB"]:
                            hasPreposision = True
                            break
                    if not hasPreposision:
                        continue

                    finloc = self._findPhraseEnd(deploc, depRes)
                    deploc,finloc = self._getWholePhrase(deploc,finloc,depRes)
                    if self.isRepeat(deploc, finloc, [conjloc]):
                        continue                    

                    phrase = self._getPhrase(deploc, finloc, depRes)
                    phrases = phrase.split('@#$%^&')
                    phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                    res.append([depRes[keyloc]["dependent"], phrases, 
                                depRes[deploc]["dep"],"%s[%d]" % (depRes[fvloc]["dependent"], fvloc), 
                                "pattern_7(%s)" % depRes[deploc]["dep"], 
                                "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res

    # 过滤  暂时过滤
    def filter(self, tmpres):
        finalpres = []
        for phrases in tmpres:
            tphrases = [ph for ph in phrases[1] if 100 > len(ph) > 3]
            if len(tphrases) <= 0:
                continue
            phrases[1] = tphrases
            # if phrases[2] == "nmod:poss":  本来基本上nmod:poss都是所有格your，但是发现了一个例外The Product's meeting functionality also enables you to be seen by other participants through your built-in device camera.
            #     continue
            finalpres.append(phrases)
        return finalpres

    def parseDepRes(self, depRes):
        
        res = []

        depRes = self._formatDepRes(depRes)
        depRes = self._formatDepRes2(depRes)
        if depRes == None:
            return res
        
        keylocs = self._findWordLocs(PERM_KEYWORD_LIST, depRes)
        if len(keylocs) == 0:
            return res
        
        readyRes = set()
        for keyloc in keylocs:
            
            tmpres = self.filter(self._pattern0(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)

            tmpres = self.filter(self._pattern1(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)
            
            tmpres = self.filter(self._pattern2(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)

            tmpres = self.filter(self._pattern3(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)

            tmpres = self.filter(self._pattern4(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)

            tmpres = self.filter(self._pattern5(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)

            tmpres = self.filter(self._pattern6(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)

            tmpres = self.filter(self._pattern7(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)

        return res

    def parseSentence(self, sentence):
        
        depRes = self.depParser.parse(sentence)
        self.depRes = depRes
        return self.parseDepRes(depRes)

    pass

# find_all = lambda c, s: [x for x in range(c.find(s), len(c)) if c[x] == s]

if __name__ == "__main__":

    senParser = SentenceParser()

    pts = [
        "Navigation by your camera.",
        "Navigation by your camera.",
        "Navigation by using camera.",
        "Record image by using camera.",
        "Recording image by using camera.",
    ]

    # for idx, ts in enumerate(pts):
    #     print("%d. %s" % (idx, ts))
    #     res = senParser.parseSentence(ts)
    #     print(senParser.depParser.prettyRes(senParser.depRes))
    #     for e in res:
    #         print(e)
    #     print('\n')

    ts = "The app may request access to the app requires your permission to use device's camera, which allows the app to turn on/off camera or flash."
    print(ts + '\n')
    res = senParser.parseSentence(ts)
    print(senParser.depParser.prettyRes(senParser.depRes))
    for e in res:
        print(e)