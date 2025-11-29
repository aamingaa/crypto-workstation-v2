"""
Alpha 模型训练模块
负责多模型训练、预测、集成
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from xgboost import XGBRegressor
import lightgbm as lgb


class AlphaModelTrainer:
    """
    Alpha 模型训练器：支持多模型训练与集成
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, selected_factors):
        """
        Args:
            X_train (np.ndarray): 训练集特征
            X_test (np.ndarray): 测试集特征
            y_train (np.ndarray): 训练集标签
            y_test (np.ndarray): 测试集标签
            selected_factors (list): 因子名称列表
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.selected_factors = selected_factors
        
        self.models = {}
        self.predictions = {}
        self.ensemble_weights = None
    
    def train_all_models(self, use_normalized_label=True):
        """
        训练所有模型（OLS, Ridge, Lasso, XGBoost, LightGBM）
        
        Args:
            use_normalized_label (bool): 是否使用标准化 label（推荐True）
        
        Returns:
            self
        """
        print("正在训练模型...")
        
        # 选择训练标签（这里假设已经传入了正确的y）
        train_label = self.y_train
        test_label = self.y_test
        
        # 1. 线性回归
        print("训练线性回归模型...")
        lr_model = LinearRegression(fit_intercept=True)
        lr_model.fit(self.X_train, train_label)
        self.models['LinearRegression'] = lr_model
        print(f"  系数: {lr_model.coef_.flatten()[:5]}... (显示前5个)")
        print(f"  截距: {lr_model.intercept_}")
        
        # 2. Ridge回归
        print("训练Ridge回归模型...")
        ridge_model = Ridge(alpha=0.2, fit_intercept=True)
        ridge_model.fit(self.X_train, train_label)
        self.models['Ridge'] = ridge_model
        
        # 3. Lasso回归
        print("训练Lasso回归模型...")
        lasso_model = LassoCV(fit_intercept=True, max_iter=5000)
        lasso_model.fit(self.X_train, train_label.flatten())
        self.models['Lasso'] = lasso_model
        
        # 4. XGBoost
        print("训练XGBoost模型...")
        X_train_df = pd.DataFrame(self.X_train, columns=self.selected_factors)
        y_train_series = pd.Series(train_label.flatten())
        X_test_df = pd.DataFrame(self.X_test, columns=self.selected_factors)
        y_test_series = pd.Series(test_label.flatten())
        
        xgb_model = XGBRegressor(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            objective='reg:squarederror',
            random_state=0,
            early_stopping_rounds=20
        )
        
        xgb_model.fit(
            X_train_df, y_train_series,
            eval_set=[(X_test_df, y_test_series)],
            verbose=False
        )
        
        self.models['XGBoost'] = xgb_model
        
        # 5. LightGBM
        print("训练LightGBM模型...")
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting': 'gbdt',
            'learning_rate': 0.054,
            'max_depth': 3,
            'num_leaves': 32,
            'min_data_in_leaf': 50,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'lambda_l1': 0.05,
            'lambda_l2': 120,
            'verbose': -1
        }
        
        lgb_train = lgb.Dataset(X_train_df, y_train_series)
        lgb_val = lgb.Dataset(X_test_df, y_test_series, reference=lgb_train)
        
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=500,
            valid_sets=lgb_val,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        self.models['LightGBM'] = lgb_model
        
        print("所有模型训练完成")
        return self
    
    def make_predictions(self):
        """
        生成所有模型的预测结果（缩放到 [-5, 5]）
        
        Returns:
            self
        """
        print("正在生成预测...")
        
        for model_name, model in self.models.items():
            if model_name == 'LightGBM':
                train_pred = model.predict(self.X_train)
                test_pred = model.predict(self.X_test)
            else:
                train_pred = model.predict(self.X_train).flatten()
                test_pred = model.predict(self.X_test).flatten()
            
            # 缩放到 [-5, 5]（参考 go_model 逻辑）
            min_val = abs(np.percentile(train_pred, 99))
            max_val = abs(np.percentile(train_pred, 1))
            scale_n = 2 / (min_val + max_val) if (min_val + max_val) > 0 else 1.0
            
            train_pred_scaled = (train_pred * scale_n).clip(-5, 5)
            test_pred_scaled = (test_pred * scale_n).clip(-5, 5)
            
            self.predictions[model_name] = {
                'train': train_pred_scaled,
                'test': test_pred_scaled
            }
        
        print(f"预测生成完成，共 {len(self.predictions)} 个模型")
        return self
    
    def ensemble_models(self, weight_method='equal', backtest_fn=None):
        """
        模型集成（等权重或基于Sharpe加权）
        
        Args:
            weight_method (str): 'equal' 或 'sharpe'
            backtest_fn (callable, optional): 回测函数，用于计算Sharpe（仅weight_method='sharpe'时需要）
        
        Returns:
            self
        """
        print(f"正在构建模型集成（方法: {weight_method}）...")
        
        model_names = list(self.predictions.keys())
        
        if weight_method == 'equal':
            # 等权重
            weights = {name: 1.0/len(model_names) for name in model_names}
            print(f"使用等权重组合方式")
            
        elif weight_method == 'sharpe':
            # 基于Sharpe加权
            if backtest_fn is None:
                print("⚠️  weight_method='sharpe' 需要提供 backtest_fn，回退到等权重")
                weights = {name: 1.0/len(model_names) for name in model_names}
            else:
                print(f"正在计算基于夏普比率的权重...")
                sharpe_ratios = {}
                
                for model_name in model_names:
                    train_pos = self.predictions[model_name]['train']
                    _, train_metrics = backtest_fn(train_pos, 'train')
                    sharpe_ratios[model_name] = abs(train_metrics['Sharpe Ratio'])
                    print(f"  {model_name}: Sharpe = {train_metrics['Sharpe Ratio']:.4f}")
                
                total_sharpe = sum(sharpe_ratios.values())
                if total_sharpe > 0:
                    weights = {name: sharpe/total_sharpe for name, sharpe in sharpe_ratios.items()}
                else:
                    print("  ⚠️  所有模型夏普比率均为0，改用等权重")
                    weights = {name: 1.0/len(model_names) for name in model_names}
        else:
            raise ValueError(f"不支持的权重方法: {weight_method}")
        
        self.ensemble_weights = weights
        print("\n模型组合权重:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.2%}")
        
        # 计算加权组合预测
        all_train_preds = [self.predictions[name]['train'] for name in model_names]
        all_test_preds = [self.predictions[name]['test'] for name in model_names]
        
        ensemble_train = np.zeros_like(all_train_preds[0])
        ensemble_test = np.zeros_like(all_test_preds[0])
        
        for i, name in enumerate(model_names):
            ensemble_train += weights[name] * all_train_preds[i]
            ensemble_test += weights[name] * all_test_preds[i]
        
        self.predictions['Ensemble'] = {
            'train': ensemble_train,
            'test': ensemble_test
        }
        
        print(f"\n集成完成，共 {len(self.predictions)} 个模型（含Ensemble）")
        return self
    
    def get_predictions(self):
        """返回所有预测结果"""
        return self.predictions
    
    def get_models(self):
        """返回所有训练好的模型"""
        return self.models
