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
            1. 合并全部**相邻**compound/fixed/flat关系, 包括sub-compund
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
            elif isFixed(dep):
                
                siblinggloc = findTrueGov(idx - 1)
                if siblinggloc == gloc:
                    locMap[idx] = findTrueGov(depRes[gloc]["govloc"])
                    depRes[gloc]["dependent"] = "%s %s" % (depRes[gloc]["dependent"], dependent)
                    # 暂时全认为是介词
                    depRes[gloc]["pos"] = "IN"

            if depRes[idx]['dep'] == "compound:prt" and getPos(depRes[gloc]["pos"]) == wordnet.VERB:
                locMap[idx] = gloc
                # 更新gov
                depRes[gloc]["dependent"] = "%s %s" % (depRes[gloc]["dependent"], dependent)

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
        "We may also collect contact information for other individuals when you use the sharing and referral tools available within some of our Services to forward content or offers to your friends and associates.",
        "The Kinect microphone can enable voice chat between players during play. The microphone also enables voice commands for control of the console, game, or app, or to enter search terms.",
        "The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms. Learn more about voice data collection.",
        "Below are the information might collected by Chrome record audio: Used by Chrome. Used to give sites ability to ask users to utilize microphone and used to provide voice search feature.",
        "If you grant microphone access to the Pandora app on your device, we will receive your voice data when you interact with a voice feature on our app. You can perform a search or control Pandora with your voice by tapping on the microphone icon in the search bar, or by using the wake word \"Hey Pandora\" if you have enabled \"Listen for \'Hey Pandora\'\" in your app settings.",
        "If granted permission by a user, My Home Screen Apps uses access to a device's microphone to facilitate voice-enabled search queries and may access your deviceâs camera, photos, media, and other files",
        "Microphone Access. If granted permission by a user, we use access to a phone's microphone to facilitate voice enabled search queries. All interaction and access to the microphone is user initiated and voice queries are not shared with third party apps or providers.",
        "Image search. Image search. Authority to access photo albums, authority to access the camera, and camera authority",
        "We collect Contact Information in order to provide with the Advanced Search Functionality, and Matching Functionality, which allows a User to identify unknown and unfamiliar phone numbers and have its address book contacts automatically associated with corresponding publicly available pictures and social network IDs of such contacts.",
        "JVSTUDIOS does not save or upload your contacts. Permission to access contact information is used when you search contacts in JVSTUDIOS search bar.",
        "ONEX SOFTECH PVT LTD does not save or upload your contacts. Permission to access contact information is used when you search contacts in ONEX SOFTECH PVT LTD search bar.",
        "Finding your friends on the Service: If you choose, you can locate your friends with Gametime United accounts through our \"Find friends\" feature. The \"Find friends\" feature allows you to choose to locate friends either through your contact list, social media sites (such as Twitter or Facebook) or through a search of names and usernames on Gametime United",
        "When you search for a friend who has HeyTell, the email address, phone number, Twitter ID, or Facebook ID you search for is transmitted to HeyTell systems, where it is used to locate a matching contact. This information is used solely to find the individual contact you're trying to locate and is not stored permanently on HeyTell systems. We do not scan, request, nor store the full contents of your device's address book; HeyTell contact searches are only performed for a single contact at a time.",
        "You are also able to link your WeChat contact list with the contact lists on your device and in your account on third party services, in order to search for and connect with contacts on those contact lists who also have a WeChat account.",
        "Some features like searching a contact from the search bar require access to your Address book. This is required for the only purpose to provide the user the ability to search a contact in the Application. No data from your Address Book is collected, shared or sent in any way on the net.",
        "Contact Search: Our search function supports searching contacts in the address book to help you quickly find the phone number of your family member or friends. Once found, you can call directly. This requires the use of your address book information, including contact names, avatars, and numbers;",
        "The headset's microphones enable voice commands for navigation, controlling apps, or to enter search terms. Learn more about voice data collection.",
        "HoloLens also processes and collects data related to the HoloLens experience and device, which include cameras, microphones, and infrared sensors that enable motions and voice to navigate.",
        "we will collect the necessary information, including Location information. When you need to use certain location-based services, such as using navigation software, viewing the weather conditions in a geographic location, retrieving your phone's location, sharing your geographic location with others, etc., with your consent, we may collect and process approximate or precise information about the actual location of the equipment you use. For example: latitude and longitude information, country or area code, city code, community identifier, navigation route, operator identifier, etc. Surely, unless you provide consent or in order to comply with a legal requirement by the relevant country or region, we will not continue to collect your location information to identify your whereabouts. You have the right to turn off location service permissions for related apps directly on your mobile device.",
        "TomTom navigation products and services need location data and other information to work correctly.",
        "It pinpoints your family's exact location and provides navigational help on a map so you can route to their determined destination.",
        "As Offline Map Navigation app is a GPS based navigation application which uses your location while using the app or all the time",
        "Location information (only for specific services/functionalities): various types of information on your accurate or approximate location if you use location-related services (navigation software, weather software, and the software with device-locating functionality). For example, region, country code, city code, mobile network code, mobile country code, cell identity, longitude and latitude information, time zone settings, language settings. You can restrict access to location information of each application at any time within the phone settings (Settings, Permissions).",
        "For example, we collect your Location Data in order to provide you with navigation services and offer you with transportation options and services to your destinations.",
        "Location information: accurate or ambiguous location information we collect when you use navigation software or search weather conditions, longitude and latitude generated by GPS or Wi-Fi hotspot, community identity or country code, location area codeï¼LACï¼, trace area code(TAC), routing area code(RAC), mobile country code (MCC), mobile network code (MNC), ARFCN, UARFCN, EARFCN, etc.",
        "The app uses GPS location data for the following purposes: the Application may use your location in background mode in order to notify you about speed cameras and other road hazards when another navigation app in active mode.",
        "With navigation you are able to find the optimal route to your destination, even inside buildings. You can relive and share your location experiences with your friends on different Social Networking Services (\"SNS\").",
        " For example, navigation features in our Apps need access to your location to work properly and responses to voice queries that involve contacts' information will not work as well without knowing the spelling of your contacts.",
        "The Application uses GPS technology (or other similar technology) to determine your current location and display it on a map or during a turn-by-turn navigation.",
        "For example: we or a third party may use your location information to provide you with weather forecast push, geographic location navigation and other related information services;",
        "Some processing associated with the purpose of providing you our Services include providing you with navigation services to your parking location.",
        "Huawei will collect data about your device and how you and your device exchange information with Huawei products and services. This type of information includes Location information. Huawei will collect, use, and process the approximate or precise location of your device when you access some location-based services (for example, when you search, use navigation software, or view the weather of a specific location). Location information can be obtained based on the GPS, Wi-Fi, and service provider network ID.",
        "If the user wishes to use the Services which include location features, such as the recording of a trail or of a point of interest or the navigation through a downloaded trail, the Services may imply the processing of the location of the user, which will be used for the purposes established hereunder and to allow or improve the Service.",
        "The Service may send your Location Data to HERE when you use location enabled features of the Service, such as enable navigation, ask information about nearby services or offerings, use search features, provide you with relevant offers from Transport Providers and public transportation vendors, as well as when the Service asks for new maps for new areas you have navigated into.",
        "TomTom navigation products and services need location data and other information to work correctly.",
        "We use your location and route information to create a detailed route history of all of your journeys made when using the Service. We use this history to offer the Service to you, to improve the quality of the Service it offers to you and to all of its users, to improve the accuracy of its mapping and navigation data, and more as described in detail in theÂ Privacy Policy. This history is associated with your account. This history is private to you and is not shared in any way.",
        "We may collect your location data, for example from the GPS functionality of your mobile device or deducted from your IP Address (\"Location Data\"). This data is used by us to provide our location-based services (GPS navigation service, road planner, research of nearby points of interests)",
        "For example, when using our navigation or localization apps, we must collect your precise location, speed and bearings.",
        "We may collect your GPS location information if you use location-based apps such as maps, navigation and weather service apps. You can turn off the location sharing service of your devices to stop sharing your device's location information.",
        "Location data provided via the Licenced Application is for basic navigational purposes only",
        "Bestie only collects and uses your personal information out of the following purposes hereby specified in this policy: When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Microphone, for recording voices in the videos.",
        "We collect the following permissions and corresponding information following the criterion of minimal data collection with your consent: â· Microphone permissions: for video shooting and editingï¼icrophone, for recording voices in the videos.",
        "including your public profile, the lists you create, and photos, videos and voice recordings as accessed with your prior consent through your device's camera and microphone sensor",
        "When you shoot or edit the photos or videos, to provide you with corresponding services, we may need you to grant the permissions for the following terminals: Cameras, for shooting photos and taking videos.",
        " audio, visual, or similar information, such as photos, videos, video footage or recordings you choose to post to our Sites or provide by granting us access to your camera while using Sites or services",
        "Images recorded by cameras fitted to Sky's engineer vans",
        "Take Photos and Videos:Â This permission allows us to use your deviceâs camera to take photos / videos and turn ON/OFF Camera Flash.",
        "Camera and Video Access Permissions. This device access permission allows the Service to access your deviceâs camera and to otherwise capture images/video from the device",
        "This permission allows APPSTARSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash",
        "Camera for Android asks for CAMERA permissions is for using the camera to take photoes and record videos.",
        " Camera for Android use the RECORD_AUDIO permissions is to record audio when recording videos.",
        "Take pictures and videos:- We use this permission because the Camera app needs this permission to take pictures.",
        "Functions necessary for the services related to cloud exhibition: when you participate in an online cloud exhibition through Buyer APP, we need to activate the authority of the camera and microphone in the device you use to complete the complete video recording and the functions attached to the recording process, and secure your consent in the form of a pop-up window when you activate the authority for the first time",
        "This permission allows JVSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash. We do not save or upload your photos/videos.",
        "android.permission. RECORD_AUDIO use camera phone to take short video support feature edit and make video",
        "CAMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device",
        "we may record your image through security cameras when you visit ASUS Royal Club repair stations and ASUS offices.""58. For example, a photo editing app might access your device's camera to let you take a new photo or access photos or videos stored on your device for editing",
        "Android.hardware. CAMERA -- Use your camera to take photos, no data is collected.",
        "Camera access. Take photos and upload",
        "The app needs access to the camera to fulfill its main purpose (recording videos).",
        "If you choose to use the Karaoke function (video), we will require access to your camera and microphone in order to enable you to make karaoke video recordings over tracks on JOOX Music and allow you to share them with your connections on JOOX Music.",
        "Used for accessing the camera or capturing images and video from the device.",
        "CAMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device.",
        "Take pictures and videos absurd Labs need this permission to use your phone's camera to take pictures and videos. We never use them or upload them.",
        "By having a Vedantu account, you have explicitly given consent for us to capture images (followed by analysis), camera/mic permissions to make video calls and record the same.",
        "CAMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device.",
        "We collect the following permissions and corresponding information following the criterion of minimal data collection with your consent: â· Camera permissions: for video shootingï¼AMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device. This permission allows the Service to support three features related to phone camera Scan QR Code for establishing a wireless connection between two devices faster because this method can bypass the manual input of wireless password.",
        "We collect the following permissions and corresponding information following the criterion of minimal data collection with your consent: â· Microphone permissions: for video shooting and editingï¼ndroid.permission. RECORD_AUDIO use camera phone to take short video support feature edit and make video",
        "We access your photos to provide you a lot of photo editor features, gallery function and photo preview function normally",
        "We use your device's microphone to record only after click the recording button. The audio saved locally in your device. We do not collect or transfer any audio data to us or to third party.",
        "including your public profile, the lists you create, and photos, videos and voice recordings as accessed with your prior consent through your device's camera and microphone sensor",
        "The microphone access is required to record voice during the composing process under the following conditions: \"MIC\" button has been tapped on,",
        "Microphone: This permission is needed in the case the user wants to record the Athan audio with his own voice.",
        "in my application, I request somes sensitive permissions to make functions of application work properly `android.permission. RECORD_AUDIO: to use microphone to record voice as the main feature of the app described`",
        ". Therefore, while using the App, you may be asked to provide us with access to your microphone. This permission allows us to use the device's microphone to record audio",
        "The microphone access is required to record voice during the composing process under the following conditions: \"Record to Audio file\" item has been chosen from \"Record\" menu.",
        "Allows the App to make Recordings using the microphone.",
        "Some of our Apps offer you the option to talk to the virtual character of such Apps. All of our Apps that offer such feature will always ask you for your permission to access the microphone on your device in advance. If you decide to give us the permission, the virtual character will be able to repeat what you say to him. Please note that our Apps do not have a function to record audio, so what you say to the virtual character is not stored on our servers, it is only used by the App to repeat to you in real time. If you decide not to give us the permission to access the microphone on your device, the virtual character will not be able to repeat after you.",
        "Microphone, To enable voice command related actions",
        r"Whether accessing the DD/BR Online Services from your home computer, mobile phone, or other device, Dunkinâ Brands and its agents collect information you directly provide. For example, we collect information when you register an account, join our loyalty program (hereinafter \"Loyalty Program\", enroll in our mailing lists or text message campaigns, locate a restaurant, apply for a job, interact with Customer Care, or otherwise communicate or transact with us through the DD/BR Online Services. We also collect information when you access the DD/BR Online Services using voice functionality services available through the microphone on a device.",
        "Voice control features will only be enabled if you affirmatively activate voice controls by giving us access to the microphone on your device. You can disable our access to your voice data by turning off our access to your microphone on your device at any time by using the operating system settings on your device or by muting your microphone. See our Voice Data FAQ for more information.",
        "Samsung will collect your voice commands when you make a specific request to the Smart TV by clicking the activation button either on the remote control or on your screen or by speaking a wake word (e.g. \"Hi, Bixby\") and speaking into the microphone on the remote control or Smart TV.",
        "Microphone; for detecting your voice and command,",
        " Using your microphone for making note via voice",
        "Call recorder uses your phone's microphone and call audio source to record calls. It does not transfer any audio or voice data to us or to any third party. It can upload recording files to your account in cloud services if you use any in Premium version.",
        "The App \"Automatic Call Recorder\" records phone calls. All recorded calls are recorded on the phone using the microphone on the device.",
        "Recording Call, Microphone",
        "ACR must use your phone's microphone and call audio source to record calls.",
        "We may share your personal data with other Insight Timer users or third parties who access audio or video recordings of Services where you have participated (for example, where you have participated in a video-conference with your camera or microphone enabled that is recorded and made available for later viewing)",
        "In the past 12 months, we have collected the following categories of personal information listed in the CCPA: Audio or visual data, including your profile picture or video of your participation in the Services (for example, where your camera is enabled in videoconference sessions that are recorded for later viewing).",
        "Meeting room. Authority to access photo albums, save-to-album function, authority to access the camera, uploading attachments, downloading attachments, camera authority, and microphone authority",
        "The Product's meeting functionality also enables you to be seen by other participants through your built-in device camera.",
        "To provide certain features (e.g. online video calling), we must access your microphone, camera, or location, with your permission, as described below Microphone: We access the microphone on your device (with your permission) to record audio messages and deliver sound during video calls.",
        "By having a Vedantu account, you have explicitly given consent for us to capture images (followed by analysis), camera/mic permissions to make video calls and record the same.",
        "We access the camera on your device (with your permission) to take your profile pictures and deliver realtime images during video calls.",
        "Our keyboard Application has a built-in feature to convert voice to text in your language. The microphone is used only when this feature is active. This functionality is available only for supported languages.",
        "If you choose to use the Karaoke function, (voice only), we will require access to your microphone",
        "With your prior consent we will be allowed to use the microphone for songsâ immediate identification and lyricsâ synchronization.",
        "WHAT PERMISSIONS WE ASKED? This application take some sensitive permissions like CAMERA: To open camera in your phone.",
        "Camera use: We only request this permission if you wish to take and share a photo within the app",
        "android.permission. CAMERA We use camera phone to take photo support feature edit photo and make video.",
        "App does not collect location information from your device by default. But if you enable geotagging feature from DSLR Camera settings, app will get location in background to attach that location with photo. In either case we dont export, store or send location date to ourselfs or any third party.",
        "This permission allows APPSENCETECHNOLOGIES to use your device's camera to take photos / videos and turn ON/OFF Camera Flash. We do not save or upload your photos/videos.",
        "Camera and Video Access Permissions. This device access permission allows the Service to access your deviceâs camera and to otherwise capture images/video from the device",
        "Coinoscope mobile application uses a phone camera to capture images of coins.",
        "Camera for Android asks for CAMERA permissions is for using the camera to take photoes and record videos.",
        "Take pictures and videos:- We use this permission because the Camera app needs this permission to take pictures.",
        "android.permission. CAMERA is required to let the app take pictures",
        "To add certain content, like pictures or videos, you may allow us to access your camera or photo album",
        "Camera. It is required for Login with Eyeprint-ID, Withdraw / Deposit Cash via QR Code, Payment with Barcode and Take a Profile Photo functions.",
        "Cameras and photos: in order to be able to send and upload your photos from your camera, you must give us permission to access them.",
        "CAMERA permission so you can take photo from your phone's camera",
        "Our products may request your permission to access your smartphone's camera to take picture. The picture will only be used, modified, stored, and shared after the permission of the player. You may at any any time turn off this feature by turning it off at the device level through your settings.",
        "This permission allows JVSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash. We do not save or upload your photos/videos.",
        "Camera: This permission is used to capture pictures of the boarding point or bus before the journey. This image can then uploaded as part of multimedia reviews.",
        "For example, a photo editing app might access your device's camera to let you take a new photo or access photos or videos stored on your device for editing",
        "Android.permission. CAMERA -- Use your camera to take photos, no data is collected.",
        "Doc Scanner requests following permissions on your device: cAMERA, required to capture image of documents using device camera.",
        "This permission applies to the camera functions in boAt ProGear. When you agree to use this privilege, you can control your phone to take a picture with Smartphone. Use of this right does not result in disclosure of your personal information.",
        "CAMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device",
        "This permission allows APPSTARSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash",
        "Camera; for taking selfies and pictures using voice,",
        "Some of our app may use this permission to take photo or show live camera photo on your screen when you request.",
        "This permission help us to get an image from your Camera (take picture from Camera) for photo edit.",
        "CAMERA: Take photos by user's device camera.",
        "Motorola's Camera application allows you to take better pictures by using face detection and other analytics on what is in your viewfinder to deliver the best image, which may occur before or after you triggered the shutter",
        "We use this permission because the Camera app needs this permission to take pictures.",
        "Used for accessing the camera or capturing images and video from the device.",
        "CAMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device.",
        "In Qr code reader, the app requests some sensitive permissions to make functions of application work properly permission CAMERA: To take picture to decode QR code and barcode, this is main feature of the app.",
        "Information Stored on Your Mobile Device: With your permission, we may collect information stored on your mobile device, such as photos you post to the Service, or access resources on your mobile device, such as the camera when you decide to take a photo and post it to the Service.",
        "We access your camera to provide you camera function normally",
        "Take pictures and videos absurd Labs need this permission to use your phone's camera to take pictures and videos. We never use them or upload them.",
        "Camera permission: For capturing picture for edit album art or artist art directly through the app.",
        "CAMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device.",
        "This product is a full range of photography, editing, photo community photography products, so we need to use your camera authorization to take photos,",
        "The application does not collect accurate or real-time information on the location or cameras of the device. If the application has the option to take photos, permission will be requested and the photos are not stored on any server.",
        "Daily Yoga will require the permission to access your camera which is used for the purpose of taking photos when you voluntarily share your photos in Daily Yoga Community.",
        "Camera: Authorisation is required so that an app can use your deviceâs camera function, perhaps to take photos or capture QR codes. The relevant app will only access the camera if you use the relevant function in the app.",
        " You need to turn on the camera permission to open the phone camera and use the flash to provide you with flashlight lighting",
        "This permission allows DIY Locker to turn ON/OFF your device's camera flashlight.",
        "Permission android.permission. CAMERA to turn on Flashlight, not take a photo.",
        "This permission allows Melody Music to turn ON/OFF your device's camera flashlight.",
        "This permission allows JVSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash. We do not save or upload your photos/videos.",
        "This permission allows APPSTARSTUDIOS to use your device's camera to take photos / videos and turn ON/OFF Camera Flash",
        "This permission allows Cherish Music to turn ON/OFF your device's camera flashlight.",
        "This Application uses the camera feature & permission to support the torch light feature.",
        "Led Flashlight only use the camera feature provided by mobile device, and don't collect any infomation from user. we use CAMERA feature to open the flash of the camera, and it is used for lighting.",
        "We only access your device camera when using app features like Flashlight or getting camera resolution.",
        "Some of our Apps may request camera permissions, which is considered a \"Sensitive Permission\". This permission allows you to take photos/videos and turn ON/OFF Camera Flash. If you do not wish to share your photos and videos, you can choose to deny such access.",
        "This permission allows Solo Launcher to turn ON/OFF your device's camera flashlight.",
        "Due to requirements of Google Play Developer Policy to take user permissions for certain things for example, Camera Permission for the Flash on Call App for our Mobile Applications, we may ask users to allow these permissions in order for the apps to run some specific features. As stated earlier, we do not collect any Personal Information from our users.",
        "The only purpose using this permission is to use camera LED as flashlight.",
        "This permission allows ONEX SOFTECH PVT LTD to use your device's camera to take photos / videos and turn ON/OFF Camera Flash. We do not save or upload your photos/videos.",
        "Explaination for permissions Take camera permission: Simulate a transparent screen and need to use the camera's camera permissions.",
        "Depending on your personal device and App permission settings, when using the App, we may collect or have access to your camera. When enabled, this may allow the App to access the camera to scan and input payment method details.",
        "Camera: Tiny Scanner needs this permission to use camera to scan docs.",
        "Use your phone camera to scan the QR code to download our mobile app",
        "Camera. It is required for Login with Eyeprint-ID, Withdraw / Deposit Cash via QR Code, Payment with Barcode and Take a Profile Photo functions.",
        "We (and our service providers) may collect personal information (such as your name and email address, telephone number, payment information and preferences) from you when you scan a QR code using your camera indicating the type of content you would like to access.",
        "We will only apply to the users for permissions when necessary to ensure the normal use of the functions. There are specific sensitive permissions as follows camera access permission(for users to turn on the camera to scan the barcodes to add tracking numbers for quick tracking)",
        "Doc Scanner requests following permissions on your device: cAMERA, required to capture image of documents using device camera.",
        "For the TraceTogether App to offer the value-added feature of scanning SafeEntry QR codes, the App needs to use your deviceâs camera.",
        "To operate, Fast Scanner requires the following permissions: Camera: using phone's camera to scan document.",
        "Camera: To allow you to scan check-in QR codes for contact tracing purposes, or set up your profile picture",
        "Camera to scan barcodes in different formats to add passes",
        "INFORMATION COLLECTED we may collect the following information: your Phone's Camera so that you can search products by scanning the barcode as well as capture photos of products that your reviewing on our platform.",
        "Camera: Authorisation is required so that an app can use your deviceâs camera function, perhaps to take photos or capture QR codes. The relevant app will only access the camera if you use the relevant function in the app.",
        "CAMERA. Used for permissions that are associated with accessing camera or capturing images and videos from the device. This permission allows the Service to support three features related to phone camera Scan QR Code for establishing a wireless connection between two devices faster because this method can bypass the manual input of wireless password.",
        "n order to offer some pretty cool features, we require you to authorize us to use your system resources when installing iReader, these authorizations and applications are as follows camera. iReader provides the \"scan\" function which can quickly find out whether the books you like have electronic versions in iReader store.",
        "Camera: Simple Scanner needs this permission to use camera to scan docs.",
        "Pic2shop accesses your device's camera to scan barcodes. All the image processing occurs on the device, no image captured by Pic2shop ever leaves your device.",
        "Scan QR code on the homepage. Scan QR code on the homepage. Authority to access photo albums and camera authority""177. Camera. It is required for Login with Eyeprint-ID, Withdraw / Deposit Cash via QR Code, Payment with Barcode and Take a Profile Photo functions.",
        "If you choose, cameras can be used to sign you in automatically using your iris.",
        "For this purpose, a photo (\"selfie\") is saved when your face is recorded using your device's camera",
        "If you choose, the camera can be used to sign in to the Xbox network automatically using facial recognition",
        "In case you agree to opt for KYC verification as available on our Services, depending on the device camera permission we will request you to click in real time an image of your face and an image of your identity proof in quick succession in order to verify your identity.",
        "We may apply facial recognition technology to your security photo to facilitate camera-enabled rapid embarkation, debarkation, at entry and exit of the vessel at ports of call",
        "In order to provide the Services, we use face recognition technology to recognize faces in photos and camera experiences",
        "When you first use the face recognition function and upload your facial photos to the App, the App will pop up a popup window to provide this Privacy Policy for you and ask for your agreeing. You may choose to grant us the permission to access to your camera and album to obtain your facial photo in your mobile device.",
        "If you choose, the camera can be used to sign you in to the Xbox network automatically using facial recognition. This data stays on the console and is not shared with anyone, and you can choose to delete this data from your console at any time.",
        "we may also use information from Apple's TrueDepth camera to improve the quality of our augmented reality experiences.",
        "In addition, we have and rely on a legitimate interest in using your Personal Data as follows to provide AR experiences. In order to do this, Niantic needs to locally derive AR geospatial data from your device camera.",
        "To improve the quality of our augmented reality experiences, information from the TrueDepth camera is used in real-time, we don't store this information on our servers or share it with third parties.",
        "Camera Permission: This permission is needed to change the album art on the user's command.",
        "We access your photos to provide you a lot of photo editor features, gallery function and photo preview function normally",
        "you may voluntarily grant us the permission to access to the camera or photo album to obtain photo stored in your mobile device.",
        "A list of your contacts, including names and email addresses, if you upload individuals contacts in order to make referrals and generate referral bonuses. We use this information to offer our Services to individual contacts you designate, including referral bonuses.",
        "We collect Contact Information in order to provide with the Advanced Search Functionality, and Matching Functionality, which allows a User to identify unknown and unfamiliar phone numbers and have its address book contacts automatically associated with corresponding publicly available pictures and social network IDs of such contacts.",
        "We may use your contact data, such as any data that you voluntarily provide to us like your phone number, email address, social media account handles, contacts, or contacts list, to authenticate your account, to communicate with you, to connect you with friends or other users, to invite your friends to iFunny's services at your request,",
        "Some of our Sites also allow users to invite friends to participate in activities by providing their friends' contact details or importing contacts from your address book or from other sites.",
        "Before inviting your contacts, BuzzBreak will and is required to ask your permission to access the contact list. With your permission, you will be able to view and invite your contacts to use BuzzBreak from within the app.",
        "Network details. If you decide to invite new members to join Nextdoor, you can choose to share their residential or email address with us, or share your contacts with us, so we can send an invitation and follow-up reminders to potential new members on your behalf.",
        "Some of our Sites also allow users to invite friends to participate in activities by providing their friendsâ contact details or importing contacts from your address book or from other sites.",
        "When you use our Games, we may solicit your permission to access your contacts list (e.g., address book) so you can be matched with individuals from your contacts list who participate in our Games and you can invite your friends to play.",
        " This may include your username, password, email address, phone number, age, gender, contacts (when you invite friends from third party applications, such as Facebook or your mobile phone's contacts application), and contact list.",
        "The following permissions are taken on app level that we require from you: phone state/Address book access is required to help users contact any Islamic organization and invite their friends and family to use our app respectively.",
        "We may use your contact data, such as any data that you voluntarily provide to us like your phone number, email address, social media account handles, contacts, or contacts list, to authenticate your account, to communicate with you, to connect you with friends or other users, to invite your friends to America's best pics and video's services at your request, to respond to communications sent to you by us, to keep records of our communication, or to pursue or defend against legal claims.",
        "Some of our Sites also allow you to invite your friends to participate in activities by providing their contact details or importing contacts from your address book or from other sites.",
        "This Application may use the Personal Data provided to allow Users to invite their friends, for example through the address book, if access has been provided, and to suggest friends or connections inside it.",
        "We collect the following types of information about you: Inviting a third party to use our Service, You have the option to sync your contacts list on your mobile device with the Service so that you can invite your contacts to join the Service via email or text message",
        "If you choose to use our invitation service to invite a third party to the Service through our \"Invite friends\" or \"GT Assistant\" features, you may directly choose a friend to invite through your mobile device's native contact list.",
        "Data provided by users to Quick Ride includes user's Contacts List provided user to choose to refer his contacts for referrals / invites.",
        "We use data about you (such as your profile, profiles you have viewed or data provided through address book uploads or partner integrations) to help others find your profile, suggest connections for you and others (e.g. Members who share your contacts or job experiences) and enable you to invite others to become a Member and connect with you.",
        "If you wish to invite your friends and contacts to use the Services, we will give you the option of either entering in their contact information manually",
        "Our automatic algorithms also use this data to suggest which of your contacts you might like to communicate with on Marco Polo. We use this information to suggest people to invite to join Marco Polo but we do not contact anyone in your phoneâs contact list without your permission",
        "Some of our Sites also allow users to invite friends to participate in activities by providing their friends' contact details or importing contacts from your address book or from other sites.",
        "The app accesses the contacts on the phone in order to display the contacts and so the app can record or ignore calls from specific contacts.",
        "We ask your permission before syncing your contacts.",
        "In addition, unless you opt-out (which you may do at any time in the SoLive application), your Profile is discoverable by other SoLive users, including by way of example in listings of contacts that will include your proximity to other users.",
        "We display all the phone calls in the phone list in the form of lists, and you can dial the phone directly in the message center. In order to make you use this function normally, we need to get call record information to implement the display and edit management of the call record list, including the telephone number, the way to call (the incoming calls, the unanswered calls, the dialed and rejected calls), and the time of the call. Meanwhile, in order to help you quickly select contacts in dialing, we need to read your address book contact information.",
        "Address Book Information. We collect information from your device's address book, which may include the names and contact information of individuals contained in your address book (collectively, \"Address Book Information\"). Currently, we use Address Book Information for the purpose of enabling you to send invitations and messages, to build a list of your contacts, and to notify you if one of your contacts registers with the App.",
        "display the name of the contact as it appears in your address book when a call is received on the Service, and sync your contacts with Viber running on Windows, MacOS, Linux, Android tablets, iPads and Windows Tablets",
        "The Licensor requests and the End User provides the following data: access to the User's contact list for the purpose of showing the progress of the User's friends, as well as the User's name and email address for signup and authentication purposes.",
        "In addition, if you permit us to do so, the App may access your device's address book solely in order to add someone to your contacts.",
        " for the Android version, we ask for permission to access your contact details/profile on your mobile device, so that we can add or find your Guardian account on your phone.",
        "Specifically, we provide the following features: Whether to add contacts automatically using the address book in your device",
        "We offer an \"Auto Add Friends\" feature which automatically adds other users as your friends in our Services when you upload information of your friend in your address book on your device. We will access only the phone numbers registered in your mobile device's address book only when you have enabled this feature.",
        "You can use the Add from Contacts feature and provide us, if permitted by applicable laws, with the phone numbers in your address book on a regular basis, including those of users of our Services and your other contacts",
        "Personal data is any information from or about an identified or identifiable person, including information that Zoom can associate with an individual person. We may collect, or process on behalf of our customers, the following categories of personal data when you use or interact with Zoom Products: contacts and Calendar Integrations: Contact information added by accounts or their users to create contact lists on Zoom, which may include contact information a user integrates from a third-party app.",
        "Upon providing your consent which will be obtained during the registration/application process, you understand that Dhani shall have the right to access and store the following: your SMS; contact list in your directory; call history; location; and device information to determine your eligibility and enhance your credit limit, if applicable.",
        "We collect information you choose to upload, sync or import from a device (such as an address book or call log or SMS log history), which we use for things like helping you and others find people you may know.",
        "or more information on our ratings feature, please read the note below justdial will use your mobile number and contacts list to identify friends in your network who have rated establishments and are users of Justdial Services.",
        " Personal Information also includes information about you or others that may be accessed by our system directly from your Device, including from your address book, location, photos or contacts folder, in order to enable certain features of an Application or the Services, such as the feature that finds and suggests mutual friends and other individuals who you may know",
        "read the contact information of the address book to find the corresponding contact quickly",
        "For instance, when you provide your contact list for finding friends on the Services, we delete the list after it is used for adding contacts as friends.",
        "We use your personal data to find matches based on your contact information and your address book",
        "Certain of our Services give you the ability to import your address book contacts, including names, e-mail addresses, and social media handles, if available, or manually enter them so that you can find your contacts on FGFF and invite them to play our apps or to other of our Services.",
        "With your permission, Teams will sync your device contacts periodically and check for other Teams users that match contacts in your deviceâs address book.",
        "Justdial may periodically access your contact list and address book on your mobile device to find and keep track of mobile phone numbers of other users of the Service.",
        "For further information on sharing your Facebook contact list with us, please see Find other users and invite your friends.",
        "This Application may request access to your address book (Example, for sharing the app to your friends).",
        "You may also choose to share your mobile deviceâs Contacts list with us (including names, numbers, emails, Facebook ID, Apple ID, etc) and we will store it on our servers to help you better improve our service, e.g. connecting with friends on our service.",
        "We use data about you (such as your profile, profiles you have viewed or data provided through address book uploads or partner integrations) to help others find your profile, suggest connections for you and others (e.g. Members who share your contacts or job experiences) and enable you to invite others to become a Member and connect with you.",
        "With your permission, Teams will sync your device contacts periodically and check for other Teams users that match contacts in your deviceâs address book.",
        "In addition, when you install the Service on your device and register with CHAMET, you may will be asked to allow us access to your address book. If you consent, we will have access to contact information in your address book on the device you use for the Service (names, numbers, emails, and Facebook IDs, but not notes or other personal information in your address book) and we will store them on our servers and use them to help you use the Service, for example, by synchronizing your CHAMET contacts between different devices you may want to use with the Service.",
        "We receive and store any information you knowingly enter on the Services, whether via computer, mobile phone, other wireless device, or that you provide to us in any other way. With your express consent, you may be able to upload, import or sync contact information from your mobile device (for example, from your address book) to Remind.",
        "Data collected and for which purposes: If made available by Drupe in your territory and by your operating system, you can choose to share with us the names, numbers and e-mail addresses contained in your device's address book (\"Contact Information\"), for the purpose of providing the Caller Identification Functionality",
        "Financial Information: We and our third party payment providers will have access to your preferred payment method and billing address details at the time of your making payments or receiving payments, as may be applicable",
        "For instance if payment is required for any of our Services, we will request your credit card or other payment information.",
        "We collect data necessary to process your payment if you make purchases, such as your payment instrument number (such as a credit card number), and the security code associated with your payment instrument.",
    ]

    # for idx, ts in enumerate(pts):
    #     print("%d. %s" % (idx, ts))
    #     res = senParser.parseSentence(ts)
    #     # print(senParser.depParser.prettyRes(senParser.depRes))
    #     for e in res:
    #         print(e)
    #     print('\n')

    ts = "ACR must use your phone's microphone and call audio source to record calls."
    print(ts + '\n')
    res = senParser.parseSentence(ts)
    print(senParser.depParser.prettyRes(senParser.depRes))
    for e in res:
        print(e)