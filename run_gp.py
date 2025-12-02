from pathlib import Path
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parent
    # 确保本地项目根目录优先于环境同名第三方包（如 utils）
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    pkg_dir = project_root / "gp_crypto_next"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))

    from gp_crypto_next.main_gp_new import GPAnalyzer

    # yaml_file_path = project_root / "gp_crypto_next" / "parameters.yaml"

    yaml_file_path = project_root / "gp_crypto_next" / "coarse_grain_parameters.yaml"

    analyzer = GPAnalyzer(str(yaml_file_path))
    # analyzer.run()

    # # Option2 - 直接评估现有因子库中的所有因子， 执行metric打分 （可以执行另外的一组metric，重新定义另一个metric_dict即可）。不需要运行gplearn.
    # analyzer.evaluate_existing_factors()

    ## Option3 - 寻找出优秀的因子，并绘制出滚动夏普和pnl曲线,再加工模型模型
    # analyzer.read_and_cal_metrics()
    exp_pool = analyzer.elite_factors_further_process()
    pos_test,pos_train = analyzer.go_model(exp_pool)
    analyzer.real_trading_simulation_plot(pos_test,pos_train,0.000)

if __name__ == "__main__":
    main()


