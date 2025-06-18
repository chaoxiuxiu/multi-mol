import re
import io
import argparse
from tqdm import tqdm
import multiprocessing

# 规范化tokenizer
def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub('\s+', '', smi) == ''.join(tokens)
    except:
        return ''
    return ' '.join(tokens)

# detokenize
def detokenizer(smi):
    return re.sub('\s+', '', smi)


def main(args):
    input_fn = args.fn
    # 每次读一行
    def lines():
        with io.open(input_fn, 'r', encoding='utf8', newline='\n') as srcf:
            for line in srcf:
                yield line.strip()
    #
    results = []
    # 得到总数量
    total = len(io.open(input_fn, 'r', encoding='utf8', newline='\n').readlines())
    # 放入线程池
    pool = multiprocessing.Pool(args.workers)

    if args.tokenize:
        for res in tqdm(pool.imap(smi_tokenizer, lines(), chunksize=100000), total=total):
            if res:
                results.append('{}\n'.format(res))
    else:
        for res in tqdm(pool.imap(detokenizer, lines(), chunksize=100000), total=total):
            if res:
                results.append('{}\n'.format(res))

    if args.output_fn is None:
        output_fn = '{}.bpe'.format(input_fn)
    else:
        output_fn = args.output_fn
    io.open(output_fn, 'w', encoding='utf8', newline='\n').writelines(results)
    print('{}/{}'.format(len(results), total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fn', type=str)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output-fn', type=str, default=None)
    # # 如需将detokenizer使用到这里，False
    parser.add_argument('--tokenize', type=bool, default=True)
    args = parser.parse_args()
    main(args)