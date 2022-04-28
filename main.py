# Author: Yu-Cheng Tsai
# Student ID: 0716074
# HW ID: hw2
# Due Date: 2022/04/21

from re import S
import spacy
import pandas as pd
import numpy as np
SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["pobj", "dobj", "dative", "attr", "oprd"]
nlp = spacy.load("en_core_web_trf")
# spacy.require_gpu()


def getSubsFromConjunctions(subs):
    """Get subjectives which 

    Args:
        subs (list): current list of collected subjectives

    Returns:
        moreSubs: extend subjectives connecting with conjunction
    """
    moreSubs = []
    for sub in subs:
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps or "or" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ in [
                            "NOUN", "PRON", "PROPN"]])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs


def getObjsFromConjunctions(objs):
    """Get objectives which 

    Args:
        objs (list): current list of collected objectives

    Returns:
        moreObjs: extend objectives connecting with conjunction
    """
    moreObjs = []
    for obj in objs:
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps or "or" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ in [
                            "NOUN", "PRON", "PROPN"]])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs


def getVerbsFromConjunctions(verbs):
    """Get verbs which 

    Args:
        verbs (list): current list of collected objectives

    Returns:
        moreVerbs: extend verbs connecting with conjunction
    """
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps or "or" in rightDeps:
            moreVerbs.extend(
                [tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    # print(moreVerbs)
    return moreVerbs


def findSubs(tok):
    """Find the subjectives with given verb.

    Args:
        tok (spacy.tokens.token.Token): _description_

    Returns:
        list: list of found subjectives

    """
    subs = []
    head = tok.head
    # 往前找到第一個動詞或名詞
    while head.pos_ not in ["VERB", "AUX"] and head.pos_ not in ["NOUN", "PRON", "PROPN"] and head.head != head:
        head = head.head
    # 如果是動詞或助動詞
    if head.pos_ in ["VERB", "AUX"]:
        # 看前面有沒有關代真正的主詞(Relcl)
        subs.extend(getSubFromRelcl(head))
        # 看前面有沒有(SUBJECTS)
        subs.extend([tok for tok in head.lefts if tok.dep_ in SUBJECTS])
        # 如果有找到主詞，做主詞的延展
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            # print("subs? ", head.text, subs)
            return subs, verbNegated
        # 沒有找到且未結束，繼續遞迴找主詞
        elif head.head != head:
            return findSubs(head)
    # 如果是名詞家族，代表找到
    elif head.pos_ in ["NOUN", "PRON", "PROPN"]:
        return [head], isNegated(tok)
    return [], False


def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False


def getObjsFromPrepositions(deps):
    """Find objectives from given prepositions

    Args:
        deps (list): list of prepositions

    Returns:
        list: list of found objectives
    """
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend(
                [tok for tok in dep.rights if tok.dep_ in OBJECTS or (tok.pos_ == "PRON")])
    return objs


def getObjFromXCompOrCComp(deps):
    """Extend objectives from given `XComp` which is a kinf of `dep_`.

    Args:
        deps (list): list of verbs and auxs

    Returns:
        list: list of found objectives
    """
    for dep in deps:
        #print(dep.pos_, dep.dep_)
        if dep.pos_ in ["VERB", "AUX"] and (dep.dep_ in ["xcomp"]):
            v = dep
            rights = list(v.rights)
            #print("getObjFromXCompOrCComp", v)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            return v, objs
    return None, []


def getObjFromAgent(deps):
    """Extend objectives from given `agent` which is a kinf of `dep_`.

    Args:
        deps (list): list of `pos_` is `ADP` and `dep_` is `agent`

    Returns:
        list: list of found objectives
    """
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "agent":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            if len(objs) > 0:
                return objs
    return []


def getSubFromRelcl(v):
    """Find subjectives from given verbs with `dep_` is `relcl`.

    Args:
        v (spacy.tokens.token.Token): verb

    Returns:
        list: list of found subjectives
    """
    subs = []
    if v.dep_ == 'relcl' and v.head.pos_ in ["NOUN", "PRON", "PROPN"]:
        subs.append(v.head)
    return subs


def getAllSubs(v):
    """Find all subjectives from given verbs.

    Args:
        v (spacy.tokens.token.Token): verb

    Returns:
        list: list of found subjectives
    """
    verbNegated = isNegated(v)
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS]
    subs.extend(getSubFromRelcl(v))
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated


def getAllObjs(v):
    """Find all objectives from given verbs.

    Args:
        v (spacy.tokens.token.Token): verb

    Returns:
        list: list of found objectives
    """
    #print("getAllObjs: ", v)
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))
    potentialNewObjs = getObjFromAgent(rights)
    if len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
    #print("rights: ", rights)
    potentialNewVerb, potentialNewObjs = getObjFromXCompOrCComp(rights)
    #print("$GET XCComp: ", potentialNewVerb, potentialNewObjs)
    if len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
    elif potentialNewVerb is not None:
        _, obj = getAllObjs(potentialNewVerb)
        objs.extend(obj)

    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    else:
        moreVerbs = getVerbsFromConjunctions([v])
        for mv in moreVerbs:
            _, obj = getAllObjs(mv)
            objs.extend(obj)
    return v, objs


def findSVOs(tokens):
    """Find all key (S,V,O) tuples from given tokens of sentence.

    Args:
        toks (spacy.tokens.doc.Doc): tokens of sentence

    Returns:
        list: list of (S,V,O) tuples
    """
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" or (
        tok.pos_ == "AUX" and tok.head.pos_ != "VERB")]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            for sub in subs:
                for obj in objs:
                    # objNegated = isNegated(obj)
                    verbNegated, objNegated = False, False
                    svos.append(
                        (sub.lower_, "!" + v.lower_ if verbNegated or objNegated else v.lower_, obj.lower_))
    return svos


def printDeps(toks):
    """Print all dep information from given tokens.

    Args:
        toks (spacy.tokens.doc.Doc): tokens of sentence

    """
    for tok in toks:
        print(tok.orth_, tok.dep_, tok.pos_, tok.head.orth_, [
              t.orth_ for t in tok.lefts], [t.orth_ for t in tok.rights])


def checkSVO(s, v, o, svos):
    """Check given s, v, o whether they are same as our key (s,v,o) tuples

    Args:
        s (str): given subjective to check
        v (str): given verb to check
        o (str): given objective to check
        svos (list): list of key (s,v,o) tuples

    Returns:
        bool: `True` if given s, v, o are correct, else `False`
    """
    s_list, v_list, o_list = s.lower().split(
        ' '), v.lower().split(' '), o.lower().split(' ')
    svos = np.array(svos)
    for svo in svos:
        # print(svos[:,0], svos[:,1], svos[:,2])
        # print(svo, s_list, v_list, o_list)
        notHeadAndTailList = ['and', 'or']
        for s in notHeadAndTailList:
            if s in [s_list[0], o_list[0], v_list[0], s_list[-1], o_list[-1], v_list[-1]]:
                return False
        if (svo[0] in s_list) and (svo[1] in v_list) and (svo[2] in o_list):
            realVO_in_dataS = len([s for s in s_list if (
                s in svos[:, 1] or s in svos[:, 2]) and s not in svos[:, 0]]) == 0
            # realV_in_dataV = len([v for v in v_list if v in svos[:,1]]) == 1
            realSV_in_dataO = len([o for o in o_list if (
                o in svos[:, 0] or o in svos[:, 1]) and o not in svos[:, 2]]) == 0
            if realVO_in_dataS and realSV_in_dataO:
                return True
    return False


def RulesCheck(S, V, O, sentence):
    """Check given s, v, o whether they are (s,v,o) of given sentence

    Args:
        S (str): given subjective to check
        V (str): given verb to check
        O (str): given objective to check
        sentence (str): given sentence

    Returns:
        bool: `True` if given s, v, o are correct, else `False`
    """
    tokens = nlp(sentence)
    svos = findSVOs(tokens)
    adj_list = [t.text for t in tokens if t.pos_ in [
        'ADJ', 'NOUN', 'PROPN', 'PRON']]
    # printDeps(tokens)
    # print(svos)
    timeStrList = []
    for ent in tokens.ents:
        if ent.label_ in ['DATE', 'TIME', 'CARDINAL', 'MONEY', 'PERCENT', 'ORDINAL', 'QUANTITY']:
            timeStrList.append(ent.text)
    haveTimeStr = len([s for s in timeStrList if s in O]
                      ) == len(O) or (O[-1] in timeStrList)
    for adj in adj_list:
        if adj in V.split(" "):
            return False
    if (checkSVO(S, V, O, svos) and not haveTimeStr):
        return True
    else:
        return False


def kaggle(filename, printOpt):
    """Deal with kaggle dataset and write the submission

    Args:
        filename (str): filename of dataset
        printOpt (bool): whether to print s,v,o which is correct
    """
    ans, debug_ans = pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(filename)
    count = 0
    for _, rows in df.iterrows():
        label = RulesCheck(rows.S, rows.V, rows.O, rows.sentence)
        if label == True:
            count += 1
            if printOpt:
                print(rows.S, "|", rows.V, "|", rows.O)
        df = pd.DataFrame({'id': [rows.id], 'label': [
                          0 if label == False else 1]})
        debug_df = pd.DataFrame({'id': [rows.id], 'S': [rows.S], 'V': [rows.V], 'O': [
                                rows.O], 'label': [0 if label == False else 1]})
        ans = pd.concat([ans, df], ignore_index=True, axis=0)
        debug_ans = pd.concat([debug_ans, debug_df], ignore_index=True, axis=0)
    print(count)
    debug_ans.to_csv('debug.csv', index=False)
    ans.to_csv('submission.csv', index=False)


def test():
    # She
    # She kissed me and hugged me
    # She kissed and hugged me
    # She is attracted by me
    sentence = "A Spanish official , who had just finished a siesta and seemed not the least bit tense , offered what he believed to be a perfectly reasonable explanation for why the portable facilities were n't in service ."
    tokens = nlp(sentence)  # spacy.tokens.doc.Doc
    # print(type(tokens[0])) # spacy.tokens.token.Token
    for ent in tokens.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    svos = findSVOs(tokens)
    res = RulesCheck('Katzenberg', 'is feuding with',
                     'former employer Disney', sentence)
    print(svos, res)


def main():
    # kaggle('data.csv', printOpt=False)
    test()


if __name__ == "__main__":
    main()
