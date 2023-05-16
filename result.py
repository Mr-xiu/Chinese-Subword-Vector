def get_result(test_path='data/pku_sim_test.txt', svd_path='data/svd_result.txt', sgns_path='data/sgns_result.txt', result_path='data/total_result.txt'):
    """
    :param test_path: 测试文件的位置
    :param svd_path: svd计算余弦值结果文件的路径
    :param sgns_path: sgns计算余弦值结果文件的路径
    :param result_path: 存放结果文件的路径
    """
    with open(test_path, 'r', encoding='UTF-8') as f:
        test_lines = f.readlines()
        f.close()
    with open(svd_path, 'r', encoding='UTF-8') as f:
        svd_lines = f.readlines()
        f.close()
    with open(sgns_path, 'r', encoding='UTF-8') as f:
        sgns_lines = f.readlines()
        f.close()

    f = open(result_path, 'w', encoding='UTF-8')
    for i in range(len(sgns_lines)):
        line_svd = svd_lines[i].strip('\n').split('\t')
        line_sgns = sgns_lines[i].strip('\n').split('\t')
        if len(line_sgns) == 0 or len(line_svd) == 0:
            continue
        sim_svd = line_svd[2]
        sim_sgns = line_sgns[2]
        f.write(f'{test_lines[i][:-1]}\t{sim_svd}\t{sim_sgns}\n')
        print(f'{test_lines[i][:-1]}\t{sim_svd}\t{sim_sgns}')
    f.close()


if __name__ == '__main__':
    get_result()
