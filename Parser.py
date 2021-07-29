import nltk
from nltk.corpus.reader.wordnet import VERB
from nltk.parse import corenlp
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

PERM_KEYWORD_LIST = ["contact", "address book",
                     "camera", "microphone", "record_audio",
                     "location", "longitude", "latitude", "GPS",
                     "SMS", "phone"]

PATTERN_1_DEP_LIST = ["obl"]
PATTERN_1V_DEP_LIST = ["obl", "nmod", "dep"]
# PATTERN_1_IN_BLACK_LIST = ["in", "on", "from", "with"]
PATTERN_1_IN_BLACK_LIST = []

PATTERN_2_DEP_LIST = ["xcomp", "nsubj:pass"]

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
            
            locMap[idx] = delta
            if idx != gloc:
                # 删除节点
                del(depRes[idx - delta])
                delta += 1    
        
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
                break

        return None

    def _pattern2(self, keyloc, depRes):
        """
           句式: verb + PI to {do SCENE}

           pattern: verb {obj} PI to SCENE; verb {xcomp} SCENE
           case: Use your camera to take photos, no data is collected.

           pattern PI {nsubj:pass} verb to SCENE; verb {xcomp} SCENE
           case: CAMERA is required to let the app take pictures.
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
                    if dep not in PATTERN_2_DEP_LIST:
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
                                depRes[deploc]["dep"],"%s[%d]" % (depRes[fvloc]["dependent"], fvloc), 
                                "pattern_2(%s)" % depRes[deploc]["dep"], 
                                "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res

    def _pattern1(self, keyloc, depRes):
        """
            pattern: [SCENE\] <IN case> [PI]. [PI] {obl} [SCENE].
            case: Image recorded by your camera.
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)
        
        for conjloc in conjlocs:

            dep = depRes[conjloc]["dep"].split(':')[0]
            if dep not in PATTERN_1_DEP_LIST:
                continue

            deploc = depRes[conjloc]["govloc"]
            # 判断 govloc 和 deploc 之间是否存在介词
            hasPreposision = False
            for loc in range(deploc + 1, conjloc):
                if depRes[loc]["dep"] == "case" and depRes[loc]["pos"] == "IN" and \
                   depRes[loc]["dependent"] not in PATTERN_1_IN_BLACK_LIST:
                    hasPreposision = True
                    break
            if not hasPreposision:
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
                        "pattern_1(%s)" % depRes[conjloc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

        return res

    def _pattern1V(self, keyloc, depRes):
        """
            pattern: [SCENE] <IN case> [use PI]. [SCENE] {obl} [use].
            case: Using your microphone for making note via voice.

            pattern: [PI] <IN case> [SCENE]. [SCENE] {nmod\dep} [PI].
            case: Using your microphone for making note via voice.
        """
        res = []
        conjlocs = self._findConjWord(keyloc, depRes)
        conjlocs.append(keyloc)
        
        for conjloc in conjlocs:
            vloc = self._findDirectVerb(conjloc, depRes)
            if vloc:
                conjlocs.append(vloc)

        for conjloc in conjlocs:
            
            deplocs = []
            deplocs.extend(self._findTargetDepWord(conjloc, depRes, PATTERN_1V_DEP_LIST))

            for deploc in deplocs:

                if deploc in conjlocs:
                    continue

                # 判断 govloc 和 deploc 之间是否存在介词
                hasPreposision = False
                for loc in range(conjloc + 1, deploc):
                    if depRes[loc]["dep"] == "case" and depRes[loc]["pos"] == "IN" and \
                       depRes[loc]["dependent"] not in PATTERN_1_IN_BLACK_LIST:
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
                phrase = self._getPhrase(tdeploc, finloc, depRes)
                phrases = phrase.split('@#$%^&')
                phrases = [i.strip() for i in phrases if(len(str(i.strip()))!=0)]
                res.append([depRes[keyloc]["dependent"], phrases, 
                            depRes[deploc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc), 
                            "pattern_1v(%s)" % depRes[deploc]["dep"], "%s[%d]" % (depRes[conjloc]["dependent"], conjloc)])

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
        if depRes == None:
            return res
        
        keylocs = self._findWordLocs(PERM_KEYWORD_LIST, depRes)
        if len(keylocs) == 0:
            return res
        
        readyRes = set()
        for keyloc in keylocs:
            
            tmpres = self.filter(self._pattern1(keyloc, depRes))
            for e in tmpres:
                hashe = hashTuple(e)
                if hashe not in readyRes:
                    res.append(e)
                    readyRes.add(hashe)
            
            tmpres = self.filter(self._pattern1V(keyloc, depRes))
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

        return res

    def parseSentence(self, sentence):
        
        depRes = self.depParser.parse(sentence)
        self.depRes = depRes
        return self.parseDepRes(depRes)

    pass

# find_all = lambda c, s: [x for x in range(c.find(s), len(c)) if c[x] == s]

if __name__ == "__main__":

    senParser = SentenceParser()

    # Pattern1 test
    p1s = [
        "Image recorded by your camera.",
        "Using your microphone for making note via voice.",
        "Microphone, for recording voices in the videos.",
        "Microphone: for recording voices in the videos.",
        "Microphone; for recording voices in the videos.",
        "Microphone; for detecting your voice and command,",
        "Microphone; for detecting your voice and command",
        "Images recorded by cameras fitted to Sky's engineer vans.",
        "Microphone; for detecting your voice and command.",
        "Using microphone permissions for video shooting and editing",
        "The app needs access to the camera for recording videos.",
        "CAMERA permission so you can take photo from your phone's camera.",
        "With your prior consent we will be allowed to use the microphone for songs immediate identification and lyrics synchronization.",
        "for example, where your camera is enabled in videoconference sessions that are recorded for later viewing.",
        "The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms.",
        "Cameras and photos: in order to be able to send and upload your photos from your camera, you must give us permission to access them.",
        "we may record your image through security cameras when you visit ASUS Royal Club repair stations and ASUS offices.",
        "The Kinect microphone can enable voice chat between players during play.", # err
    ]

    p2s = [
        "If granted permission by a user, My Home Screen Apps uses access to a device's microphone to facilitate voice-enabled search queries and may access your devices camera, photos, media, and other files.",
        # v + PI to do <use> -> obj 
        "For example: we or a third party may use your location information to provide you with weather forecast push, geographic location navigation and other related information services;",
        "This permission allows APPSTARSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash.",
        "This permission allows JVSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash. We do not save or upload your photos/videos.",
        "Call recorder uses your phone's microphone and call audio source to record calls. It does not transfer any audio or voice data to us or to any third party. It can upload recording files to your account in cloud services if you use any in Premium version.",
        "android.permission. RECORD_AUDIO use camera phone to take short video support feature edit and make video",
        "For example, a photo editing app might access your device's camera to let you take a new photo or access photos or videos stored on your device for editing.",
        "CAMERA -- Use your camera to take photos, no data is collected.",
        "Coinoscope mobile application uses a phone camera to capture images of coins.",
        "Camera for Android asks for CAMERA permissions is for using the camera to take photoes and record videos.",
        # PI + v[pass] to do <nsubj:pass>
        "The microphone access is required to record voice during the composing process under the following conditions: \"MIC\" button has been tapped on.",
        "CAMERA is required to let the app take pictures",
    ]

    pts = [
        "If you wish to invite your friends and contacts to use the Services, we will give you the option of either entering in their contact information manually.",
        "The app accesses the contacts on the phone in order to display the contacts and so the app can record or ignore calls from specific contacts.",
        "We display all the phone calls in the phone list in the form of lists, and you can dial the phone directly in the message center. In order to make you use this function normally, we need to get call record information to implement the display and edit management of the call record list, including the telephone number, the way to call (the incoming calls, the unanswered calls, the dialed and rejected calls), and the time of the call. Meanwhile, in order to help you quickly select contacts in dialing, we need to read your address book contact information.",
        "The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms. Learn more about voice data collection.",
        "If granted permission by a user, My Home Screen Apps uses access to a device's microphone to facilitate voice-enabled search queries and may access your deviceâs camera, photos, media, and other files.",
        "We may also collect contact information for other individuals when you use the sharing and referral tools available within some of our Services to forward content or offers to your friends and associates.",
        "We collect the following permissions and corresponding information following the criterion of minimal data collection with your consent: Microphone permissions: for video shooting and editing.",
        "Take pictures and videos absurd Labs need this permission to use your phone's camera to take pictures and videos.",
        "This permission allows JVSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash.",
        "The microphone also enables voice commands for control of the console, game, or app, or to enter search terms.",
        "When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Cameras, for shooting photos and taking videos.",
        "We access the microphone on your device (with your permission) to record audio messages and deliver sound during video calls.",
        "With your prior consent we will be allowed to use the microphone for songs immediate identification and lyrics synchronization.",
        "TomTom navigation products and services need location data and other information to work correctly.",
        "Some of our Apps offer you the option to talk to the virtual character of such Apps. All of our Apps that offer such feature will always ask you for your permission to access the microphone on your device in advance. If you decide to give us the permission, the virtual character will be able to repeat what you say to him. Please note that our Apps do not have a function to record audio, so what you say to the virtual character is not stored on our servers, it is only used by the App to repeat to you in real time. If you decide not to give us the permission to access the microphone on your device, the virtual character will not be able to repeat after you.",
        "The Product's meeting functionality also enables you to be seen by other participants through your built-in device camera.",
        "Cameras and photos: in order to be able to send and upload your photos from your camera, you must give us permission to access them.",
        "Images recorded by cameras fitted to Sky's engineer vans.",
        "The app needs access to the camera to fulfill recording videos.",
        "If granted permission by a user, we use access to a phone's microphone to facilitate voice enabled search queries. All interaction and access to the microphone is user initiated and voice queries are not shared with third party apps or providers.",
        "Some features like searching a contact from the search bar require access to your Address book.",
        "including your public profile, the lists you create, and photos, videos and voice recordings as accessed with your prior consent through your device's camera and microphone sensor.",
        "Voice control features will only be enabled if you affirmatively activate voice controls by giving us access to the microphone on your device.",
        "The app needs access to the camera to fulfill recording videos.",
        "Some features like searching a contact from the search bar require access to your Address book.",
        "When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Microphone, for recording voices in the videos.",
        "We collect the following permissions and corresponding information following the criterion of minimal data collection with your consent: â· Microphone permissions: for video shooting and editingï¼",
        "Recording Call, Microphone",
        "Used for accessing the camera or capturing images and video from the device.",
        "The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms.",
        "HoloLens also processes and collects data related to the HoloLens experience and device, which include cameras, microphones, and infrared sensors that enable motions and voice to navigate.",
        "If you wish to invite your friends and contacts to use the Services, we will give you the option of either entering in their contact information manually.",
        "As Offline Map Navigation app is a GPS based navigation application which uses your location while using the app or all the time.",
        "including your public profile, the lists you create, and photos, videos and voice recordings as accessed with your prior consent through your device's camera and microphone sensor",
        "Permission to access contact information is used when you search contacts in JVSTUDIOS search bar.",
        "Used to give sites ability to ask users to utilize microphone and used to provide voice search feature.",
        "Some features like searching a contact from the search bar require access to your Address book.",
        "Some features like searching a contact from the search bar require access to your Address book.",
        "Camera; for taking selfies and pictures using voice.",
        "For example, when using our navigation or localization apps, we must collect your precise location, speed and bearings.",
        "About Us Studios And Locations Educating Consumers Playtest",
        "We display all the phone calls in the phone list in the form of lists",
        "We use the categories of information for the business and commercial purposes outlined here we use information to protect our company and constituents.\u00c2 We use contact, demographic, and site usage information to protect our company and customers.",
        "You have the option to request your friends to join the Services by providing their contact information.",
        "We record calls for the purpose of monitoring our call handlers and providing appropriate training for them and to keep an accurate record of what was said during a telephone conversation in the event of further issues or complaint.",
        "The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms."
    ]

    for ts in p2s:
    # for ts in pts:
        res = senParser.parseSentence(ts)
        print(ts)
        # print(senParser.depParser.prettyRes(senParser.depRes))
        for e in res:
            print(e)
        print('\n')

    # use access[NN] + PI to do 
    # ts = "If granted permission by a user, My Home Screen Apps uses access to a device's microphone to facilitate voice-enabled search queries and may access your devices camera, photos, media, and other files."
    # v + PI to do <use> -> obj 
    ts = "For example: we or a third party may use your location information to provide you with weather forecast push, geographic location navigation and other related information services;"
    ts = "This permission allows APPSTARSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash."
    ts = "This permission allows JVSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash. We do not save or upload your photos/videos."
    ts = "RECORD_AUDIO use camera phone to take short video support feature edit and make video"
    ts = "For example, a photo editing app might access your device's camera to let you take a new photo or access photos or videos stored on your device for editing."
    ts = "CAMERA -- Use your camera to take photos, no data is collected."
    ts = "Coinoscope mobile application uses a phone camera to capture images of coins."
    ts = "Camera for Android asks for CAMERA permissions is for using the camera to take photoes and record videos."
    # PI + v[pass] to do <nsubj:pass>
    ts = "The microphone access is required to record voice during the composing process under the following conditions: \"MIC\" button has been tapped on."
    ts = "CAMERA is required to let the app take pictures"


    # ts = "Our search function supports searching contacts in the address book to help you quickly find the phone number of your family member or friends. Once found, you can call directly."
    # print(ts + '\n')
    # res = senParser.parseSentence(ts)
    # print(senParser.depParser.prettyRes(senParser.depRes))
    # for e in res:
    #     print(e)