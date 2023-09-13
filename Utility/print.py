def print_result(result):
    print("method:", result['method'])
    print("dataset:", result['dataset'])
    print("subDataset:", result['subDataset'])
    print("result: [")
    if isinstance(result['result'], list):  # 检查 "result" 字段是否是一个列表
        for entry in result['result']:
            print("    {")
            if isinstance(entry, dict):  # 检查列表中的每个元素是否是字典
                for key, value in entry.items():
                    print(f"        {key}: {value},")
                print("    },")
            else:
                print(f"        Error: {entry},")  # 如果不是字典，打印错误信息
                print("    },")
    else:
        print(f"    Error: {result['result']},")  # 如果 "result" 不是列表，打印错误信息
    print("]")
    print("==================================================")
    print()
    print()
