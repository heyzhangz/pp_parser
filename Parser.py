import nltk
import json
from nltk.tokenize import WordPunctTokenizer
from nltk.parse import corenlp
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

PERM_KEYWORD_LIST = ["contact", "address book",
                     "camera", "microphone", "record_audio",
                     "location", "longitude", "latitude", "GPS",
                     "SMS", "phone"]

PATTERN_1_DEP_LIST = ["obl", "appos"]
PATTERN_2_DEP_LIST = ["advcl", "xcomp", "obl", "ccomp", "obj", "nsubj"]
PATTERN_3_DEP_LIST = ["nmod", "obl"]

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
            phrase = self._getPhrase(deploc, finloc, depRes)

            res.append([depRes[keyloc]["dependent"], phrase, 
                        depRes[conjloc]["dep"], depRes[conjloc]["dependent"], 
                        "pattern_1", depRes[conjloc]["dependent"]])

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

            case(ccomp): Some features like searching a contact from the search bar require access to your Address book.
            11 (require, VBP) ccomp  [4](searching, VBG)
            16 (book, NN)     nmod   [12](access, NN)

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
            fvlocs = self._parseFinVerb(conjloc, depRes)
            if len(fvlocs) == 0:
                continue

            for fvloc in fvlocs:
                deplocs = self._parseDepWord(fvloc, depRes)
                if len(deplocs) == 0:
                    continue
                    
                for deploc in deplocs:

                    if deploc in fvlocs or deploc in conjlocs:
                        continue
                    
                    dep = depRes[deploc]["dep"].split(':')[0]
                    if dep not in PATTERN_2_DEP_LIST:
                        continue

                    finloc = self._findPhraseEnd(deploc, depRes)
                    phrase = self._getPhrase(deploc, finloc, depRes)
                    # PI, scene, findep, finverb, patterns
                    res.append([depRes[keyloc]["dependent"], phrase, 
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

                dep = depRes[deploc]["dep"].split(':')[0]
                if not dep in PATTERN_3_DEP_LIST:
                    continue
                
                finloc = self._findPhraseEnd(deploc, depRes)
                phrase = self._getPhrase(deploc, finloc, depRes)

                res.append([depRes[keyloc]["dependent"], phrase, 
                            depRes[deploc]["dep"], depRes[conjloc]["dependent"], 
                            "pattern_3", depRes[conjloc]["dependent"]])

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
            res.extend(tmpres)

            tmpres = self._pattern2(keyloc, depRes)
            res.extend(tmpres)

            tmpres = self._pattern3(keyloc, depRes)
            res.extend(tmpres)

        return res

    def parseSentence(self, sentence):
        
        depRes = self.depParser.parse(sentence)
        self.depRes = depRes
        return self.parseDepRes(depRes)

    pass

if __name__ == "__main__":

    senParser = SentenceParser()    
    # ts = r"we may record your image through security cameras when you visit ASUS Royal Club repair stations and ASUS offices."
    # ts = r"Images recorded by cameras fitted to Sky's engineer vans."
    # ts = r"Permission to access contact information is used when you search contacts in JVSTUDIOS search bar."
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

    # res = senParser.parseSentence(ts)
    
    # print(senParser.depParser.prettyRes(senParser.depRes))
    # for e in res:
    #     print(e)

    # ds = r"As Offline Map Navigation app is a GPS based navigation application which uses your location while using the app or all the time."
    # depParser = DepParser()
    
    # res = depParser.parse(ds)
    # print(depParser.prettyRes(res))

    # with open(r"./sentences.json", 'r', encoding="utf-8") as f:
    # # with open(r"./test.json", 'r', encoding="utf-8") as f:
    #     allSens = json.load(f)

    # with open(r"./dep_sentence.txt", 'w', encoding="utf-8") as f:
    #     index = 0
    #     resDict = []
    #     for section in allSens:
    #         for item in section:
    #             sentence = item["sentences"]
    #             scene = item["view"]
    #             pi = item["privacy"]

    #             res = senParser.parseSentence(sentence)
    #             parseRes = senParser.depParser.prettyRes(senParser.depRes)
        
    #             resDict.append({
    #                 "scene": scene,
    #                 "PI": pi,
    #                 "sentence": sentence,
    #                 "parse": res,
    #                 "dep": parseRes
    #             })

    #             # write
    #             f.write("%s. %s\n" % (str(index), sentence))
    #             f.write("\n")
    #             f.write("%s -> %s\n" % (str(pi), str(scene)))
    #             f.write("\n")
    #             for e in res:
    #                 # f.write("%s -> %s, %s, %s, %s" % (e[0], e[1], e[2], e[3], e[4]))
    #                 f.write("%s -> %s,\t%s,\t%s,\t%s\t%s\n" % tuple(e))
    #             f.write("\n")
    #             f.write(parseRes)
    #             f.write("\n=====================\n\n")
    #             index += 1

            
    
    # with open(r"./dep_sentence.json", 'w', encoding="utf-8") as f:
    #     json.dump(resDict, f, indent=4)
