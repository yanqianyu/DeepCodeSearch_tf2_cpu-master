import random

def parseInput(sent):
    return [z for z in sent.split(' ')]


def toNum(data, vocab_to_int):
    # 转为编号表示
    res = []
    for z in parseInput(data):
        res.append(str(vocab_to_int.get(z, vocab_to_int["<UNK>"])))
    return res


def getVocabForAST(self, asts, vocab_size):
    vocab = set()
    counts = {}

    vocab_to_int = {}
    int_to_vocab = {}

    for ast in asts:
        for node in ast:
            if "type" in node.keys():
                counts[node["type"]] = counts.get(node["type"], 0) + 1
                vocab.update([node["type"]])
            # naive方法中path包括value
            if "value" in node.keys():
                counts[node["value"]] = counts.get(counts[node["value"]], 0) + 1
                vocab.update(node["value"])

    _sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])
    for i, word in enumerate(["<PAD>", "<UNK>", "<START>", "<STOP>"] + _sorted):
        if vocab_size is not None and i > vocab_size:
            break

        vocab_to_int[word] = i
        int_to_vocab[i] = word

    return vocab_to_int, int_to_vocab


def dfsSimplify(ast, root, path, totalpath):
    # 深度遍历 得到多条路径
    if "children" in ast[root["index"]].keys():
        if len(ast[root["index"]]["children"]) > 1:
            path.append(root["type"])
            for child in root["children"]:
                dfsSimplify(ast, ast[child], path, totalpath)
                path.pop()
        else:
            # 只有一个子节点 略过
            dfsSimplify(ast, ast[root["children"][0]], path, totalpath)

    else:
        path.append(root["value"])
        # 叶节点内容包含在path中
        totalpath.append(' '.join(path))
        return


def getNPathSimplify(ast, n):
    # 随机得到n条路径
    path = []
    totalpath = []
    dfsSimplify(ast, ast[0], path, totalpath)
    nPath = []
    for i in range(n):
        a = random.randint(0, len(totalpath) - 1)
        b = random.randint(0, len(totalpath) - 1)
        sent = ' '.join(reversed(totalpath[a].split(' ')[1:])) + ' ' + totalpath[b]
        nPath.append(sent)
    return nPath


def getPathSimplify(asts, pathNum, ast_vocab_to_int):
    # 每次训练路径都是随机抽取的
    astPathNum = [] # 所有ast的所有path的编号表示 三维数组
    for ast in asts:
        nPath = getNPathSimplify(ast, pathNum)  # 针对每个ast的n条路径
        nPathNum = []
        for path in nPath:  #每条path的编号表示
            nPathNum.append(toNum(path, ast_vocab_to_int))
        astPathNum.append(nPathNum)
        # sbt = ' '.join(getSBT(ast, ast[0]))  # 得到李戈的sbt树
    return astPathNum