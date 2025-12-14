from pathlib import Path
import sys
from multi_model_strategy.backtest_engine import BacktestEngine
from multi_model_strategy.diagnostics import DiagnosticTools


def main() -> None:
    project_root = Path(__file__).resolve().parent
    # 确保本地项目根目录优先于环境同名第三方包（如 utils）
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    pkg_dir = project_root / "gp_crypto_next"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))

    from gp_crypto_next.main_gp_new import GPAnalyzer

    yaml_file_path = project_root / "gp_crypto_next" / "coarse_grain_parameters.yaml"

    analyzer = GPAnalyzer(str(yaml_file_path))
    
    exp_pool = analyzer.read_and_pick()

    diag = analyzer.build_diagnostic_tools_from_exp_pool(
        exp_pool=exp_pool,
        fees_rate=0.0005
    )

    print("可用因子列表：")
    print(diag.selected_factors[:10])

    # 可视化因子与价格
    diag.visualize_factor_vs_price(
        factor_name="Liq_Zscore",  # 替换为实际因子名
        data_range='test',
        price_type='close',
        save_dir=analyzer.total_factor_file_dir / "diagnostics"
    )

if __name__ == "__main__":
    main()