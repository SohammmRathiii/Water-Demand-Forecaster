import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import pickle
import json
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from prophet import Prophet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastExplainer:
    
    def __init__(self, zone_id: str = None, region_type: str = 'zone'):
 
        self.zone_id = zone_id
        self.region_type = region_type
        self.feature_impacts = {}
        self.recent_forecast = None
        self.baseline_demand = None
        
    def analyze_prophet_components(self, prophet_model: 'WaterDemandProphetModel') -> Dict:

        if not prophet_model.is_trained:
            raise ValueError("Prophet model must be trained first")
        
        components = {
            'trend': prophet_model.model.trend.mean() if hasattr(prophet_model.model, 'trend') else 0,
            'yearly_seasonality': self._extract_seasonality(prophet_model, 'yearly'),
            'weekly_seasonality': self._extract_seasonality(prophet_model, 'weekly'),
            'daily_seasonality': self._extract_seasonality(prophet_model, 'daily'),
        }
        
        return components
    
    @staticmethod
    def _extract_seasonality(prophet_model: 'WaterDemandProphetModel', period: str) -> float:
        try:
            forecast = prophet_model.model.make_future_dataframe(periods=365)
            components = prophet_model.model.predict(forecast)
            
            if period == 'yearly':
                component = components.get('yearly', pd.Series([0])).mean()
            elif period == 'weekly':
                component = components.get('weekly', pd.Series([0])).mean()
            elif period == 'daily':
                component = components.get('daily', pd.Series([0])).mean()
            else:
                component = 0
            
            return float(component)
        except:
            return 0.0
    
    def calculate_lstm_feature_importance(
        self,
        lstm_model: 'WaterDemandLSTMModel',
        recent_data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:

        if not lstm_model.is_trained:
            raise ValueError("LSTM model must be trained first")
        
        X_input = tf.convert_to_tensor(
            recent_data.reshape(1, lstm_model.lookback, lstm_model.feature_dim),
            dtype=tf.float32
        )
        
        importance_scores = {}
        
        for feature_idx in range(min(len(feature_names), lstm_model.feature_dim)):
            with tf.GradientTape() as tape:
                X_input_var = tf.Variable(X_input)
                tape.watch(X_input_var)
                
                predictions = lstm_model.model(X_input_var)
                loss = tf.reduce_mean(predictions)

            gradients = tape.gradient(loss, X_input_var)
            
            feature_importance = tf.reduce_mean(
                tf.abs(gradients[:, :, feature_idx])
            ).numpy()
            
            importance_scores[feature_names[feature_idx]] = float(feature_importance)
  
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: (v/total)*100 for k, v in importance_scores.items()}
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def generate_plain_english_explanation(
        self,
        forecast_value: float,
        baseline_value: float,
        feature_importance: Dict[str, float],
        prophet_components: Dict = None,
        aggregation_level: str = 'zone'
    ) -> str:
        change_mld = forecast_value - baseline_value
        change_pct = (change_mld / baseline_value * 100) if baseline_value > 0 else 0
        
        explanation_parts = []
  
        if change_mld > 0:
            explanation_parts.append(
                f"Demand is forecast to INCREASE by {change_mld:.1f} MLD "
                f"({change_pct:+.1f}%) compared to typical {aggregation_level}-level demand."
            )
        else:
            explanation_parts.append(
                f"Demand is forecast to DECREASE by {abs(change_mld):.1f} MLD "
                f"({change_pct:.1f}%) compared to typical {aggregation_level}-level demand."
            )
        
        explanation_parts.append("")
        explanation_parts.append("KEY DRIVERS:")
        explanation_parts.append("─" * 50)
 
        top_features = dict(list(feature_importance.items())[:3])
        for feature, importance in top_features.items():
            impact = self._feature_to_impact_description(feature, importance)
            explanation_parts.append(f"• {impact}")
        
        if prophet_components:
            explanation_parts.append("")
            explanation_parts.append("SEASONAL PATTERNS:")
            explanation_parts.append("─" * 50)
            
            if abs(prophet_components.get('yearly_seasonality', 0)) > 10:
                yearly = prophet_components['yearly_seasonality']
                season_desc = "summer peak" if yearly > 0 else "winter low"
                explanation_parts.append(f"• Annual cycle: {season_desc} ({abs(yearly):.0f} MLD effect)")
            
            if abs(prophet_components.get('weekly_seasonality', 0)) > 5:
                weekly = prophet_components['weekly_seasonality']
                explanation_parts.append(f"• Weekly pattern: Weekdays vs weekends ({abs(weekly):.0f} MLD effect)")
        
        if change_mld > 100:
            explanation_parts.append("")
            explanation_parts.append("⚠️  ALERT: Large increase detected")
            explanation_parts.append("   → May require advance supply adjustments")
            explanation_parts.append("   → Check weather forecast & event calendar")
        
        return "\n".join(explanation_parts)
    
    @staticmethod
    def _feature_to_impact_description(feature: str, importance: float) -> str:
        feature_descriptions = {
            'temperature': f"Temperature effect causes {importance:.0f}% of change",
            'rainfall_mm': f"Rainfall patterns contribute {importance:.0f}% of variation",
            'hour_sin': f"Daily cycle (morning/evening peaks) drives {importance:.0f}% of change",
            'hour_cos': f"Diurnal variation accounts for {importance:.0f}% of demand shift",
            'dow_sin': f"Weekly cycle (weekday vs weekend) causes {importance:.0f}% fluctuation",
            'dow_cos': f"Day-of-week effects drive {importance:.0f}% of variance",
            'is_weekend': f"Weekend behavior contributes {importance:.0f}% to change",
            'is_holiday': f"Holiday & festival activity causes {importance:.0f}% increase",
            'is_monsoon': f"Monsoon season effects drive {importance:.0f}% of demand",
            'supply_demand_ratio': f"Supply constraints account for {importance:.0f}% of change",
            'lag1h': f"Recent demand (1h ago) influences {importance:.0f}% of forecast",
            'lag24h': f"Yesterday's pattern drives {importance:.0f}% of today's demand",
        }
        
        return feature_descriptions.get(
            feature,
            f"{feature.replace('_', ' ')} contributes {importance:.0f}%"
        )
    
    def generate_multi_level_explanation(
        self,
        city_forecast: float,
        city_baseline: float,
        zone_forecast: float,
        zone_baseline: float,
        ward_forecast: float = None,
        ward_baseline: float = None,
        feature_importance: Dict[str, float] = None
    ) -> Dict[str, str]:

        if feature_importance is None:
            feature_importance = {}
        
        return {
            'city_level': self.generate_plain_english_explanation(
                city_forecast,
                city_baseline,
                feature_importance,
                aggregation_level='city'
            ),
            'zone_level': self.generate_plain_english_explanation(
                zone_forecast,
                zone_baseline,
                feature_importance,
                aggregation_level='zone'
            ),
            'ward_level': (
                self.generate_plain_english_explanation(
                    ward_forecast,
                    ward_baseline,
                    feature_importance,
                    aggregation_level='ward'
                ) if ward_forecast is not None else None
            ),
            'recommendation': self._generate_recommendation(
                city_forecast,
                city_baseline,
                zone_forecast,
                zone_baseline
            )
        }
    
    @staticmethod
    def _generate_recommendation(
        city_forecast: float,
        city_baseline: float,
        zone_forecast: float,
        zone_baseline: float
    ) -> str:

        city_change = city_forecast - city_baseline
        zone_change = zone_forecast - zone_baseline
        
        recommendations = []
        
        if city_change > 150:
            recommendations.append("CITY-WIDE ACTION REQUIRED")
            recommendations.append("• Activate contingency supply from backup reservoirs")
            recommendations.append("• Issue public conservation notice")
            recommendations.append("• Brief media on demand situation")
        
        if zone_change > 50:
            recommendations.append(f"ZONE ALERT: {zone_change:.0f} MLD above baseline")
            recommendations.append("• Coordinate with zone superintendent")
            recommendations.append("• Prepare pressure management plan")
            recommendations.append("• Monitor supply infrastructure")
        
        if city_change < -100:
            recommendations.append("OPPORTUNITY: Demand lower than expected")
            recommendations.append("• Opportunity for maintenance activities")
            recommendations.append("• Good time for reservoir filling")
            recommendations.append("• Consider water transfer to other regions")
        
        if not recommendations:
            recommendations.append("Status: Within expected range")
            recommendations.append("• Continue normal operations")
            recommendations.append("• Monitor ongoing trends")
        
        return "\n".join(recommendations)

class WaterDemandProphetModel:
    def __init__(self, zone_id: str, interval_width: float = 0.95):

        self.zone_id = zone_id
        self.interval_width = interval_width
        self.model = None
        self.is_trained = False
        self.train_rmse = None
        self.train_mape = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:

        prophet_df = df.copy()
        prophet_df['ds'] = pd.to_datetime(prophet_df['timestamp'])
        prophet_df['y'] = prophet_df['demand_mld']
        
        prophet_df['cap'] = 2500  
        prophet_df['floor'] = 100  
        
        return prophet_df[['ds', 'y', 'cap', 'floor']]
    
    def train(self, df: pd.DataFrame, holdout_days: int = 30) -> Dict:
  
        logger.info(f"Training Prophet for {self.zone_id} ({len(df)} records)")
        
     
        prophet_df = self.prepare_data(df)
       
        split_date = prophet_df['ds'].max() - timedelta(days=holdout_days)
        train_df = prophet_df[prophet_df['ds'] <= split_date].copy()
        val_df = prophet_df[prophet_df['ds'] > split_date].copy()
        

        self.model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            interval_width=self.interval_width,
            changepoint_prior_scale=0.05, 
            seasonality_prior_scale=10.0,   
            seasonality_mode='additive',   
            growth='linear'
        )
        
        holidays_df = self._get_mumbai_holidays()
        self.model.add_country_holidays('IN')  
        
  
        self.model.fit(train_df)
 
        forecast_val = self.model.make_future_dataframe(
            periods=len(val_df),
            include_history=False
        )
        forecast_val = self.model.predict(forecast_val)
        
        val_df_reset = val_df.reset_index(drop=True)
        forecast_val_reset = forecast_val.reset_index(drop=True)
        
        y_true = val_df_reset['y'].values
        y_pred = forecast_val_reset['yhat'].values
        
        # Calculate metrics
        self.train_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        self.train_mape = float(mean_absolute_percentage_error(y_true, y_pred))
        
        self.is_trained = True
        
        metrics = {
            'model': 'Prophet',
            'zone_id': self.zone_id,
            'training_records': len(train_df),
            'validation_records': len(val_df),
            'rmse': self.train_rmse,
            'mape': self.train_mape,
            'interval_width': self.interval_width
        }
        
        logger.info(f"Prophet training complete: RMSE={self.train_rmse:.2f}, MAPE={self.train_mape:.2%}")
        return metrics
    
    def forecast(self, periods: int = 180) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        logger.info(f"Generating {periods}-day forecast for {self.zone_id}")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        forecast = forecast.tail(periods).copy()
        forecast['yhat'] = forecast['yhat'].clip(lower=100)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=100)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=100)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
    def get_components(self) -> Dict:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        components = self.model.params
        
        future = self.model.make_future_dataframe(periods=7)
        forecast = self.model.predict(future)
        
        return {
            'trend': forecast[['ds', 'trend']].tail(7).to_dict('records'),
            'yearly': forecast[['ds', 'yearly']].tail(7).to_dict('records'),
            'weekly': forecast[['ds', 'weekly']].tail(7).to_dict('records'),
        }
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
    
    @staticmethod
    def _get_mumbai_holidays() -> pd.DataFrame:
        holidays = pd.DataFrame({
            'holiday': [
                'New Year', 'Republic Day', 'Holi', 'Diwali',
                'Ganesh Chaturthi', 'Christmas', 'Summer Break'
            ],
            'ds': pd.to_datetime([
                '2026-01-01', '2026-01-26', '2026-03-15', '2026-11-08',
                '2026-08-15', '2026-12-25', '2026-05-01'
            ]),
            'lower_window': -1,
            'upper_window': 1
        })
        return holidays

class WaterDemandLSTMModel:
    
    def __init__(
        self,
        zone_id: str,
        lookback: int = 168,  # 7 days of hourly data
        forecast_horizon: int = 24,  # Forecast 1 day ahead
        feature_dim: int = 12
    ):

        self.zone_id = zone_id
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.feature_dim = feature_dim
        
        # Model components
        self.model = None
        self.lower_model = None  # For 5th percentile (lower CI)
        self.upper_model = None  # For 95th percentile (upper CI)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.is_trained = False
        self.train_history = {}
        self.train_metrics = {}
        
    def _build_architecture(self) -> keras.Model:
        """Build LSTM architecture for demand forecasting."""
        model = keras.Sequential([
            # Input layer
            keras.Input(shape=(self.lookback, self.feature_dim)),
            
            # LSTM layers with dropout (prevent overfitting)
            LSTM(128, activation='relu', return_sequences=True,
                 dropout=0.2),
            LSTM(64, activation='relu', return_sequences=False,
                 dropout=0.2),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            
            # Output layer: predict mean demand
            Dense(self.forecast_horizon, activation='linear')
        ])
        
        return model
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:

        data = df[feature_columns].values
        
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = [], []
        
        for i in range(len(data_scaled) - self.lookback - self.forecast_horizon):
            X.append(data_scaled[i:i + self.lookback, :])
            
            y.append(data_scaled[i + self.lookback:i + self.lookback + self.forecast_horizon, 0])
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict:
        logger.info(f"Training LSTM for {self.zone_id} ({len(df)} records)")
        
        # Prepare sequences
        X, y = self.prepare_sequences(df, feature_columns)
        logger.info(f"Created {len(X)} sequences ({X.shape})")
        
        # Build model
        self.model = self._build_architecture()
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        self.train_history = history.history
        self.is_trained = True
        
        # Calculate metrics on validation set
        val_size = int(len(X) * validation_split)
        X_val = X[-val_size:]
        y_val = y[-val_size:]
        y_pred = self.model.predict(X_val, verbose=0)
        
        # Inverse scale for metric calculation
        y_val_unscaled = self.scaler.inverse_transform(
            np.hstack([y_val, np.zeros((y_val.shape[0], self.feature_dim - 1))])
        )[:, 0]
        y_pred_unscaled = self.scaler.inverse_transform(
            np.hstack([y_pred, np.zeros((y_pred.shape[0], self.feature_dim - 1))])
        )[:, 0]
        
        rmse = float(np.sqrt(mean_squared_error(y_val_unscaled, y_pred_unscaled)))
        mape = float(mean_absolute_percentage_error(y_val_unscaled, y_pred_unscaled))
        mae = float(mean_absolute_error(y_val_unscaled, y_pred_unscaled))
        
        self.train_metrics = {
            'model': 'LSTM',
            'zone_id': self.zone_id,
            'lookback_hours': self.lookback,
            'forecast_horizon_hours': self.forecast_horizon,
            'training_sequences': len(X),
            'rmse': rmse,
            'mape': mape,
            'mae': mae,
            'epochs_trained': len(history.epoch)
        }
        
        logger.info(f"LSTM training complete: RMSE={rmse:.2f}, MAPE={mape:.2%}")
        return self.train_metrics
    
    def forecast(self, recent_data: np.ndarray) -> Dict:

        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        X_input = recent_data.reshape(1, self.lookback, self.feature_dim)
        
        y_pred = self.model.predict(X_input, verbose=0)[0]
        
        y_pred_unscaled = self.scaler.inverse_transform(
            np.hstack([y_pred.reshape(-1, 1), np.zeros((len(y_pred), self.feature_dim - 1))])
        )[:, 0]
        
        y_lower = y_pred_unscaled * 0.85  
        y_upper = y_pred_unscaled * 1.15  
        
        return {
            'forecast': y_pred_unscaled,
            'forecast_lower': y_lower,
            'forecast_upper': y_upper,
            'horizon_hours': self.forecast_horizon
        }
    
    def incremental_retrain(self, new_data: np.ndarray, new_targets: np.ndarray):
 
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        logger.info(f"Incremental retraining with {len(new_data)} new sequences")
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001), 
            loss='mse'
        )
        
        self.model.fit(
            new_data,
            new_targets,
            epochs=3,
            batch_size=16,
            verbose=0
        )
        
        logger.info("Incremental retraining complete")
    
    def save(self, filepath: str):
        self.model.save(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        self.model = keras.models.load_model(filepath)
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

class WaterDemandForecastingService:

    
    def __init__(self, zone_id: str):
        self.zone_id = zone_id
        self.lstm_model = WaterDemandLSTMModel(zone_id)
        self.prophet_model = WaterDemandProphetModel(zone_id)
        self.explainer = ForecastExplainer(zone_id=zone_id, region_type='zone')
        self.last_training_date = None
        self.feature_names = None
        self.baseline_demand = 500  
        
    def train_all_models(
        self,
        daily_df: pd.DataFrame,
        hourly_df: pd.DataFrame,
        hourly_feature_columns: List[str]
    ) -> Dict:

        logger.info(f"Training all models for {self.zone_id}")
        
        self.feature_names = hourly_feature_columns
        
        self.baseline_demand = float(daily_df['demand_mld'].mean())
        
        prophet_metrics = self.prophet_model.train(daily_df)
        
        lstm_metrics = self.lstm_model.train(
            hourly_df,
            hourly_feature_columns
        )
        
        self.last_training_date = datetime.now()
        
        return {
            'prophet': prophet_metrics,
            'lstm': lstm_metrics,
            'training_date': self.last_training_date.isoformat(),
            'baseline_demand_mld': self.baseline_demand
        }
    
    def forecast_short_term(
        self,
        recent_data: np.ndarray,
        hours_ahead: int = 24,
        include_explanation: bool = True
    ) -> Dict:

        if hours_ahead != self.lstm_model.forecast_horizon:
            logger.warning(f"LSTM trained for {self.lstm_model.forecast_horizon}h, "
                          f"but {hours_ahead}h requested. Using trained horizon.")
        
        result = self.lstm_model.forecast(recent_data)
        
        forecast_result = {
            'model': 'LSTM',
            'zone_id': self.zone_id,
            'forecast_type': 'short_term',
            'forecast_mld': result['forecast'].tolist(),
            'forecast_lower_mld': result['forecast_lower'].tolist(),
            'forecast_upper_mld': result['forecast_upper'].tolist(),
            'horizon_hours': result['horizon_hours'],
            'confidence_level': 0.95,
            'generated_at': datetime.now().isoformat()
        }
        
   
        if include_explanation and self.feature_names:

            feature_importance = self.explainer.calculate_lstm_feature_importance(
                self.lstm_model,
                recent_data,
                self.feature_names
            )

            forecast_value = float(result['forecast'][0])
            explanation = self.explainer.generate_plain_english_explanation(
                forecast_value=forecast_value,
                baseline_value=self.baseline_demand,
                feature_importance=feature_importance,
                aggregation_level='zone'
            )
            
            forecast_result['explanation'] = explanation
            forecast_result['feature_importance'] = feature_importance
        
        return forecast_result
    
    def forecast_medium_term(
        self,
        days_ahead: int = 180,
        include_explanation: bool = True
    ) -> Dict:
        forecast_df = self.prophet_model.forecast(days_ahead)
        
        medium_term_result = {
            'model': 'Prophet',
            'zone_id': self.zone_id,
            'forecast_type': 'medium_term',
            'forecast_data': forecast_df.to_dict('records'),
            'forecast_days': days_ahead,
            'confidence_level': 0.95,
            'generated_at': datetime.now().isoformat()
        }
        
        if include_explanation:
            components = self.explainer.analyze_prophet_components(self.prophet_model)

            first_forecast = float(forecast_df.iloc[0]['yhat'])
            explanation = self.explainer.generate_plain_english_explanation(
                forecast_value=first_forecast,
                baseline_value=self.baseline_demand,
                feature_importance={},
                prophet_components=components,
                aggregation_level='zone'
            )
            
            medium_term_result['prophet_components'] = components
            medium_term_result['explanation'] = explanation
            medium_term_result['trend_direction'] = self._describe_trend(forecast_df)
        
        return medium_term_result
    
    @staticmethod
    def _describe_trend(forecast_df: pd.DataFrame) -> str:

        if len(forecast_df) < 2:
            return "Insufficient data for trend analysis"
        
        first_forecast = forecast_df.iloc[0]['yhat']
        last_forecast = forecast_df.iloc[-1]['yhat']
        change = last_forecast - first_forecast
        change_pct = (change / first_forecast * 100) if first_forecast > 0 else 0
        
        if change_pct > 10:
            return f"Demand trend: INCREASING by {change_pct:.1f}% over {len(forecast_df)} days"
        elif change_pct < -10:
            return f"Demand trend: DECREASING by {abs(change_pct):.1f}% over {len(forecast_df)} days"
        else:
            return f"Demand trend: STABLE (±{abs(change_pct):.1f}% variation over {len(forecast_df)} days)"
    
    def get_forecast_explanations(self) -> Dict:
        components = self.prophet_model.get_components()
        
        return {
            'zone_id': self.zone_id,
            'prophet_components': components,
            'model_interpretability': {
                'lstm': 'Learns sequential patterns; feature importance via gradient analysis',
                'prophet': 'Trend + Seasonality + Holiday effects (decomposable & explainable)'
            }
        }
    
    def generate_multi_level_briefing(
        self,
        city_forecast: float,
        zone_forecast: float,
        ward_forecast: float = None,
        feature_importance: Dict[str, float] = None
    ) -> Dict[str, str]:
        city_baseline = self.baseline_demand  # Simplified
        zone_baseline = self.baseline_demand * 0.9  # Zones typically smaller
        ward_baseline = self.baseline_demand * 0.3 if ward_forecast else None
        
        return self.explainer.generate_multi_level_explanation(
            city_forecast=city_forecast,
            city_baseline=city_baseline,
            zone_forecast=zone_forecast,
            zone_baseline=zone_baseline,
            ward_forecast=ward_forecast,
            ward_baseline=ward_baseline,
            feature_importance=feature_importance or {}
        )
    
    def evaluate_models(
        self,
        test_df: pd.DataFrame
    ) -> Dict:
        return {
            'lstm_metrics': self.lstm_model.train_metrics,
            'prophet_metrics': {
                'rmse': self.prophet_model.train_rmse,
                'mape': self.prophet_model.train_mape
            },
            'recommendation': self._get_model_recommendation()
        }
    
    def _get_model_recommendation(self) -> str:
        lstm_mape = self.lstm_model.train_metrics.get('mape', float('inf'))
        prophet_mape = self.prophet_model.train_mape or float('inf')
        
        if lstm_mape < prophet_mape:
            return f"LSTM (MAPE: {lstm_mape:.2%}) for 1-7 day forecasts"
        else:
            return f"Prophet (MAPE: {prophet_mape:.2%}) for longer-term planning"

class ForecastEvaluator:
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
 
        mean_demand = np.mean(y_true)
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'Mean_Demand': float(mean_demand),
            'RMSE_as_pct_of_mean': float(100 * rmse / mean_demand),
            'Bias': float(np.mean(y_pred - y_true))  # Systematic over/under-forecast
        }
    
    @staticmethod
    def evaluate_confidence_intervals(
        y_true: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_upper: np.ndarray
    ) -> Dict[str, float]:

        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        width = np.mean(y_pred_upper - y_pred_lower)
        
        return {
            'coverage_rate': float(coverage),
            'expected_coverage': 0.95,
            'interval_width_mld': float(width),
            'is_wellcalibrated': abs(coverage - 0.95) < 0.05
        }
    
    @staticmethod
    def generate_report(
        metrics: Dict,
        ci_metrics: Dict
    ) -> str:

        report = f"""
        ╔════════════════════════════════════════════════════════════╗
        ║         WATER DEMAND FORECAST EVALUATION REPORT            ║
        ╚════════════════════════════════════════════════════════════╝
        
        ACCURACY METRICS:
        ─────────────────
        Mean Absolute Error (MAE):     {metrics['MAE']:.2f} MLD
          → On average, forecast is off by {metrics['MAE']:.2f} MLD
        
        Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f} MLD
          → {metrics['RMSE_as_pct_of_mean']:.1f}% of average demand
        
        Mean Absolute % Error (MAPE):   {metrics['MAPE']:.2%}
          → Forecast is {metrics['MAPE']:.2%} off on average
        
        Systematic Bias:                {metrics['Bias']:+.2f} MLD
          → {'Over-forecasting' if metrics['Bias'] > 0 else 'Under-forecasting'}
        
        CONFIDENCE INTERVAL QUALITY:
        ──────────────────────────────
        Coverage Rate:                  {ci_metrics['coverage_rate']:.1%}
          → {ci_metrics['coverage_rate']:.1%} of actual values fall within 95% CI
          → Target: 95% (interval is {'well-calibrated' if ci_metrics['is_wellcalibrated'] else 'miscalibrated'})
        
        Interval Width:                 ±{ci_metrics['interval_width_mld']/2:.2f} MLD (avg)
          → Narrow intervals = confident, Wide intervals = uncertain
        
        SUITABILITY FOR GOVERNMENT USE:
        ────────────────────────────────
        ✓ Accuracy: {'GOOD' if metrics['MAPE'] < 0.20 else 'FAIR' if metrics['MAPE'] < 0.30 else 'POOR'}
          → {metrics['MAPE']:.2%} error is {'acceptable' if metrics['MAPE'] < 0.20 else 'marginal'} for policy-making
        
        ✓ Reliability: {'GOOD' if ci_metrics['is_wellcalibrated'] else 'POOR'}
          → Confidence intervals are {'trustworthy' if ci_metrics['is_wellcalibrated'] else 'unreliable'}
        
        ✓ Actionability: Suitable for 24-48h advance shortage alerts
        """
        return report

class ScenarioSimulator:

    
    def __init__(self, zone_id: str, baseline_demand: float, available_supply: float):

        self.zone_id = zone_id
        self.baseline_demand = baseline_demand
        self.available_supply = available_supply
        self.current_scenario = None
        self.scenarios_history = []
        
    def apply_heatwave(self, num_days: int, max_temp: float = 45) -> Dict:
 
        temp_excess = max(0, max_temp - 35)  # Degrees above baseline
        demand_increase_pct = temp_excess * 0.05  # 5% per degree
        
        scenario = {
            'name': f'{num_days}-day Heatwave',
            'type': 'heatwave',
            'parameters': {
                'duration_days': num_days,
                'peak_temperature': max_temp,
                'temperature_excess': temp_excess
            },
            'demand_change_pct': demand_increase_pct,
            'demand_increase_mld': self.baseline_demand * (demand_increase_pct / 100),
            'forecast_demand': self.baseline_demand * (1 + demand_increase_pct / 100),
            'reason': f'High temperature (+{temp_excess}°C above baseline) drives cooling/bathing demand'
        }
        
        return self._analyze_scenario(scenario)
    
    def apply_rainfall_change(self, rainfall_change_pct: float) -> Dict:
  
        demand_change_pct = -(rainfall_change_pct / 20) * 3
        
        scenario = {
            'name': f'Rainfall {"Deficit" if rainfall_change_pct < 0 else "Surplus"} ({abs(rainfall_change_pct):.0f}%)',
            'type': 'rainfall',
            'parameters': {
                'rainfall_change_pct': rainfall_change_pct
            },
            'demand_change_pct': demand_change_pct,
            'demand_increase_mld': self.baseline_demand * (demand_change_pct / 100),
            'forecast_demand': self.baseline_demand * (1 + demand_change_pct / 100),
            'reason': f'{"Dry conditions increase" if rainfall_change_pct < 0 else "Wet conditions decrease"} outdoor water use'
        }
        
        return self._analyze_scenario(scenario)
    
    def apply_population_surge(self, growth_pct: float, duration_days: int = 30) -> Dict:
 
        demand_change_pct = growth_pct  # 1% population growth = ~1% demand growth
        
        scenario = {
            'name': f'Population Surge ({growth_pct:.0f}%, {duration_days} days)',
            'type': 'population',
            'parameters': {
                'growth_pct': growth_pct,
                'duration_days': duration_days
            },
            'demand_change_pct': demand_change_pct,
            'demand_increase_mld': self.baseline_demand * (demand_change_pct / 100),
            'forecast_demand': self.baseline_demand * (1 + demand_change_pct / 100),
            'reason': f'{growth_pct:.0f}% more people need water (domestic + commercial)'
        }
        
        return self._analyze_scenario(scenario)
    
    def apply_festival_overlap(self, num_festivals: int, avg_attendees: int = 50000) -> Dict:

        total_attendees = num_festivals * avg_attendees
        festival_demand_mld = (total_attendees * 200) / 1_000_000  # Convert L to MLD
        demand_change_pct = (festival_demand_mld / self.baseline_demand) * 100
        
        scenario = {
            'name': f'{num_festivals} Festivals ({total_attendees:,} attendees)',
            'type': 'festival',
            'parameters': {
                'num_festivals': num_festivals,
                'total_attendees': total_attendees,
                'avg_attendees_per_festival': avg_attendees
            },
            'demand_change_pct': demand_change_pct,
            'demand_increase_mld': festival_demand_mld,
            'forecast_demand': self.baseline_demand + festival_demand_mld,
            'reason': f'{total_attendees:,} festival attendees require water for ceremonies, cleaning, festivities'
        }
        
        return self._analyze_scenario(scenario)
    
    def apply_industrial_change(self, change_pct: float) -> Dict:
        industrial_baseline = self.baseline_demand * 0.3
        industrial_change_mld = industrial_baseline * (change_pct / 100)
        demand_change_pct = (industrial_change_mld / self.baseline_demand) * 100
        
        scenario = {
            'name': f'Industrial Activity {"Increase" if change_pct > 0 else "Decrease"} ({change_pct:+.0f}%)',
            'type': 'industrial',
            'parameters': {
                'change_pct': change_pct
            },
            'demand_change_pct': demand_change_pct,
            'demand_increase_mld': industrial_change_mld,
            'forecast_demand': self.baseline_demand + industrial_change_mld,
            'reason': f'Industrial sector demand changes by {change_pct:+.0f}% due to manufacturing activity'
        }
        
        return self._analyze_scenario(scenario)
    
    def combine_scenarios(self, scenarios: List[Dict]) -> Dict:

        total_demand_increase = sum(s['demand_increase_mld'] for s in scenarios)
        total_demand = self.baseline_demand + total_demand_increase
        combined_demand_pct = (total_demand_increase / self.baseline_demand) * 100
        
        scenario_names = [s['name'] for s in scenarios]
        
        combined = {
            'name': f'Combined: {", ".join(scenario_names)}',
            'type': 'combined',
            'individual_scenarios': scenarios,
            'demand_change_pct': combined_demand_pct,
            'demand_increase_mld': total_demand_increase,
            'forecast_demand': total_demand,
            'reason': f'Cumulative effect of {len(scenarios)} simultaneous scenarios'
        }
        
        return self._analyze_scenario(combined)
    
    def _analyze_scenario(self, scenario: Dict) -> Dict:

        forecast_demand = scenario['forecast_demand']
        
        stress_ratio = forecast_demand / self.available_supply
        stress_pct = (stress_ratio - 1) * 100 if stress_ratio > 1 else 0
        
        if stress_ratio <= 0.8:
            risk_category = 'Low'
            risk_description = 'Supply comfortable, no rationing needed'
        elif stress_ratio <= 0.95:
            risk_category = 'Low-Medium'
            risk_description = 'Supply adequate but monitor closely'
        elif stress_ratio <= 1.05:
            risk_category = 'Medium'
            risk_description = 'Supply tight, may need conservation request'
        elif stress_ratio <= 1.15:
            risk_category = 'Medium-High'
            risk_description = 'Likely shortage, rationing recommended'
        else:
            risk_category = 'High'
            risk_description = 'Severe shortage, immediate action required'
        
        deficit_mld = max(0, forecast_demand - self.available_supply)
        surplus_mld = max(0, self.available_supply - forecast_demand)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            stress_ratio,
            deficit_mld,
            scenario.get('type', 'combined')
        )
        
        scenario.update({
            'supply_available_mld': self.available_supply,
            'demand_forecast_mld': forecast_demand,
            'stress_ratio': stress_ratio,
            'stress_percentage': stress_pct,
            'risk_category': risk_category,
            'risk_description': risk_description,
            'deficit_mld': deficit_mld,
            'surplus_mld': surplus_mld,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
        self.current_scenario = scenario
        self.scenarios_history.append(scenario)
        
        return scenario
    
    @staticmethod
    def _generate_recommendations(stress_ratio: float, deficit_mld: float, scenario_type: str) -> List[str]:
 
        recommendations = []
        
        if stress_ratio <= 0.8:
            recommendations.append('✓ Status: Green - No action needed')
            recommendations.append('→ Monitor conditions and prepare contingency plans')
        
        elif stress_ratio <= 0.95:
            recommendations.append('→ Advisory: Monitor demand trends carefully')
            recommendations.append('→ Request voluntary 5-10% conservation from citizens')
            recommendations.append('→ Prepare contingency supply activation plans')
        
        elif stress_ratio <= 1.05:
            recommendations.append(f'⚠️ Warning: Expected deficit {deficit_mld:.0f} MLD')
            recommendations.append('→ Activate contingency supply (backup reservoirs, recycling)')
            recommendations.append('→ Reduce non-essential uses (garden watering, cleaning)')
            recommendations.append('→ Request 15% voluntary conservation')
        
        elif stress_ratio <= 1.15:
            recommendations.append(f'🔴 Critical: Severe deficit {deficit_mld:.0f} MLD')
            recommendations.append('→ Activate ALL contingency supplies')
            recommendations.append('→ Implement 25% mandatory rationing')
            recommendations.append('→ Prioritize: Hospitals → Homes → Industry')
            recommendations.append('→ Public communication campaign')
        
        else:
            recommendations.append(f'🚨 Emergency: Catastrophic deficit {deficit_mld:.0f} MLD')
            recommendations.append('→ Emergency rationing (40% reduction)')
            recommendations.append('→ Declare water emergency')
            recommendations.append('→ Activate all crisis measures')
            recommendations.append('→ Coordinate with neighboring water boards')
        if scenario_type == 'heatwave':
            recommendations.append('→ Heat: Increase cooling water supply from groundwater')
            recommendations.append('→ Public: Reduce shower duration and garden watering')
        
        elif scenario_type == 'rainfall':
            recommendations.append('→ Drought: Prioritize reservoir refilling')
            recommendations.append('→ Agriculture: Switch to drip irrigation')
        
        elif scenario_type == 'population':
            recommendations.append('→ Surge: Verify population data and temporary status')
            recommendations.append('→ Housing: Reduce per-capita allocation if permanent growth')
        
        elif scenario_type == 'festival':
            recommendations.append('→ Event Planning: Coordinate with festival organizers')
            recommendations.append('→ Recycling: Use grey water for festival cleaning')
        
        elif scenario_type == 'industrial':
            recommendations.append('→ Industry: Mandate water recycling and efficiency')
            recommendations.append('→ Incentives: Offer discounts for conservation')
        
        return recommendations
    
    def generate_scenario_report(self, scenario: Dict = None) -> str:
        if scenario is None:
            scenario = self.current_scenario
        
        if scenario is None:
            return "No scenario has been simulated yet."
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║         WATER DEMAND SCENARIO ANALYSIS REPORT                  ║
╠════════════════════════════════════════════════════════════════╣
║ Zone: {scenario.get('zone_id', 'Unknown')} | Date: {scenario.get('analysis_timestamp', 'N/A')}  
║
║ SCENARIO: {scenario['name']}
║ ──────────────────────────────────────────────────────────────
║
║ DEMAND ANALYSIS:
║ ───────────────
║ Baseline Demand:           {self.baseline_demand:>7.1f} MLD
║ Scenario Impact:           {scenario['demand_increase_mld']:>+7.1f} MLD ({scenario['demand_change_pct']:>+5.1f}%)
║ Forecast Demand:           {scenario['forecast_demand']:>7.1f} MLD
║
║ SUPPLY & STRESS:
║ ────────────────
║ Available Supply:          {self.available_supply:>7.1f} MLD
║ Stress Ratio:              {scenario['stress_ratio']:>7.2f}x (demand/supply)
║ Stress Percentage:         {scenario['stress_percentage']:>7.1f}%
║
║ SHORTAGE ANALYSIS:
║ ──────────────────"""
        
        if scenario['deficit_mld'] > 0:
            report += f"""
║ Water Deficit:             {scenario['deficit_mld']:>7.1f} MLD 🔴
║ Percentage Deficit:        {(scenario['deficit_mld']/scenario['forecast_demand']*100):>7.1f}%"""
        else:
            report += f"""
║ Water Surplus:             {scenario['surplus_mld']:>7.1f} MLD ✓
║ Surplus Percentage:        {(scenario['surplus_mld']/self.available_supply*100):>7.1f}%"""
        
        report += f"""
║
║ RISK ASSESSMENT:
║ ────────────────
║ Risk Category:             {scenario['risk_category']:>10s}
║ Description:               {scenario['risk_description']}
║
║ ROOT CAUSE:
║ ───────────
║ {scenario.get('reason', 'N/A')}
║
║ RECOMMENDED ACTIONS:
║ ───────────────────"""
        
        for i, rec in enumerate(scenario['recommendations'], 1):
            report += f"\n║ {i}. {rec}"
        
        report += """
║
╚════════════════════════════════════════════════════════════════╝
        """
        return report
    
    def compare_scenarios(self, scenario_list: List[Dict]) -> str:
        
        for scenario in scenario_list:
            name = scenario['name'][:20].ljust(20)
            demand = f"{scenario['forecast_demand']:.0f}".rjust(6)
            stress = f"{scenario['stress_ratio']:.2f}x".rjust(6)
            risk = scenario['risk_category'].ljust(15)
            
            if scenario['deficit_mld'] > 0:
                balance = f"-{scenario['deficit_mld']:.0f}MLD"
            else:
                balance = f"+{scenario['surplus_mld']:.0f}MLD"
            
            comparison += f"║ {name} │ {demand} │ {stress} │ {risk} │ {balance}\n"
        
        comparison += "║\n╚════════════════════════════════════════════════════════════════════════════════╝"
        return comparison

def example_training_pipeline():

    print("=" * 70)
    print("URBAN WATER INTELLIGENCE PLATFORM - FORECASTING ENGINE")
    print("=" * 70)

    print("\n[1] Loading historical data...")

    dates = pd.date_range('2023-01-01', '2025-12-31', freq='H')
    
    # Synthetic hourly demand with seasonality
    np.random.seed(42)
    hourly_demand = 500 + \
                    100 * np.sin(np.arange(len(dates)) * 2*np.pi / (24*365)) + \
                    50 * np.sin(np.arange(len(dates)) * 2*np.pi / 24) + \
                    20 * np.random.randn(len(dates))
    hourly_demand = np.clip(hourly_demand, 200, 1000)
    
    hourly_df = pd.DataFrame({
        'timestamp': dates,
        'demand_mld': hourly_demand,
        'temperature': 25 + 10*np.sin(np.arange(len(dates)) * 2*np.pi / (24*365)),
        'rainfall_mm': np.maximum(0, np.random.poisson(2, len(dates))),
        'hour_sin': np.sin(2*np.pi*dates.hour/24),
        'hour_cos': np.cos(2*np.pi*dates.hour/24),
        'dow_sin': np.sin(2*np.pi*dates.dayofweek/7),
        'dow_cos': np.cos(2*np.pi*dates.dayofweek/7),
        'is_weekend': (dates.dayofweek >= 5).astype(int),
        'is_holiday': 0,  # Simplified
        'is_monsoon': ((dates.month >= 6) & (dates.month <= 9)).astype(int),
        'supply_demand_ratio': 1.1 + 0.1*np.random.randn(len(dates)),
    })
    
    # Aggregate to daily for Prophet
    daily_df = hourly_df.groupby(hourly_df['timestamp'].dt.date).agg({
        'demand_mld': 'mean'
    }).reset_index()
    daily_df.columns = ['timestamp', 'demand_mld']
    daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])
    
    print(f"   ✓ Loaded {len(hourly_df):,} hourly records ({len(daily_df)} days)")
    
    print("\n[2] Training forecasting models...")
    
    service = WaterDemandForecastingService(zone_id='ZONE_A')
    
    feature_columns = [
        'demand_mld', 'temperature', 'rainfall_mm',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'is_holiday', 'is_monsoon', 'supply_demand_ratio',
        'hour_sin'  # Dummy to match feature_dim=12
    ]
    
    training_metrics = service.train_all_models(
        daily_df=daily_df,
        hourly_df=hourly_df,
        hourly_feature_columns=feature_columns
    )
    
    print("\n   Prophet (Medium-term):")
    print(f"     RMSE: {training_metrics['prophet']['rmse']:.2f} MLD")
    print(f"     MAPE: {training_metrics['prophet']['mape']:.2%}")
    
    print("\n   LSTM (Short-term):")
    print(f"     RMSE: {training_metrics['lstm']['rmse']:.2f} MLD")
    print(f"     MAPE: {training_metrics['lstm']['mape']:.2%}")
    
    print("\n[3] Generating forecasts WITH EXPLANATIONS...")
    
    # Short-term (LSTM)
    recent_hourly_data = hourly_df[['demand_mld', 'temperature', 'rainfall_mm',
                                      'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                                      'is_weekend', 'is_holiday', 'is_monsoon',
                                      'supply_demand_ratio', 'hour_sin']].tail(168).values
    
    short_term = service.forecast_short_term(recent_hourly_data, hours_ahead=24, include_explanation=True)
    print(f"\n   Short-term (LSTM, next 24 hours):")
    print(f"     Forecast: {short_term['forecast_mld'][0]:.2f} MLD")
    print(f"     Range: {short_term['forecast_lower_mld'][0]:.2f} - {short_term['forecast_upper_mld'][0]:.2f} MLD")
    
    # Show feature importance
    if 'feature_importance' in short_term:
        print(f"\n   What's driving this forecast?")
        for feature, importance in list(short_term['feature_importance'].items())[:3]:
            print(f"     • {feature}: {importance:.1f}% importance")
    
    # Show plain-English explanation
    if 'explanation' in short_term:
        print(f"\n   Plain-English Explanation:")
        print("   " + "\n   ".join(short_term['explanation'].split("\n")))
    
    # Medium-term (Prophet)
    medium_term = service.forecast_medium_term(days_ahead=180, include_explanation=True)
    forecast_data = medium_term['forecast_data']
    print(f"\n   Medium-term (Prophet, next 6 months):")
    print(f"     Forecast: {forecast_data[0]['yhat']:.2f} MLD")
    print(f"     Range: {forecast_data[0]['yhat_lower']:.2f} - {forecast_data[0]['yhat_upper']:.2f} MLD")
    
    if 'trend_direction' in medium_term:
        print(f"     Trend: {medium_term['trend_direction']}")
    
    if 'explanation' in medium_term:
        print(f"\n   Why this forecast?")
        print("   " + "\n   ".join(medium_term['explanation'].split("\n")[:6]))

    print("\n[4] Evaluating forecast quality...")
    
    y_true = daily_df['demand_mld'].tail(100).values
    y_pred = daily_df['demand_mld'].rolling(1).mean().tail(100).values
    
    metrics = ForecastEvaluator.calculate_metrics(y_true, y_pred)
    ci_metrics = ForecastEvaluator.evaluate_confidence_intervals(
        y_true,
        y_pred * 0.9,
        y_pred * 1.1
    )
    
    print(ForecastEvaluator.generate_report(metrics, ci_metrics))

    print("\n[5] GOVERNANCE BRIEFINGS (for different decision-makers)...")
    
    city_forecast = 600.0  # Entire Mumbai
    zone_forecast = 150.0  # ZONE_A
    ward_forecast = 50.0   # One ward
    
    briefings = service.generate_multi_level_briefing(
        city_forecast=city_forecast,
        zone_forecast=zone_forecast,
        ward_forecast=ward_forecast,
        feature_importance={
            'temperature': 35,
            'is_weekend': 25,
            'rainfall_mm': 20,
            'hour_sin': 15,
            'is_holiday': 5
        }
    )
    
    print("\n   ┌─ CITY-LEVEL BRIEFING (for Mayor & Commissioner)")
    print("   │  Strategic planning for entire Mumbai")
    print("   │  " + "─" * 50)
    city_brief = briefings['city_level'].split('\n')
    for line in city_brief[:5]:  # First 5 lines
        print(f"   │  {line}")
    
    print("\n   ├─ ZONE-LEVEL BRIEFING (for Zone Superintendent)")
    print("   │  Operational planning for ZONE_A")
    print("   │  " + "─" * 50)
    zone_brief = briefings['zone_level'].split('\n')
    for line in zone_brief[:5]:
        print(f"   │  {line}")
    
    if briefings.get('ward_level'):
        print("\n   └─ WARD-LEVEL BRIEFING (for Ward Officer)")
        print("      Community engagement & conservation")
        print("      " + "─" * 50)
        ward_brief = briefings['ward_level'].split('\n')
        for line in ward_brief[:4]:
            print(f"      {line}")
    
    print("\n   RECOMMENDATIONS:")
    print("   " + "─" * 50)
    recs = briefings['recommendation'].split('\n')
    for rec in recs:
        print(f"   {rec}")
    
    print("\n[6] SCENARIO SIMULATION - What-If Analysis...")
    print("     Planning for different conditions to stress-test supply")
    
    # Initialize scenario simulator for ZONE_A
    baseline_demand = 150.0  # MLD
    available_supply = 165.0  # MLD (10% buffer)
    
    simulator = ScenarioSimulator(
        zone_id='ZONE_A',
        baseline_demand=baseline_demand,
        available_supply=available_supply
    )
    
    print(f"\n   Baseline conditions:")
    print(f"     • Zone: ZONE_A")
    print(f"     • Typical demand: {baseline_demand} MLD")
    print(f"     • Available supply: {available_supply} MLD")
    print(f"     • Buffer: {((available_supply/baseline_demand - 1)*100):.1f}%")
    
    # Scenario 1: Heatwave
    print("\n   ─── SCENARIO 1: Heatwave (45°C for 15 days) ───")
    heatwave = simulator.apply_heatwave(num_days=15, max_temp=45)
    print(f"     Forecast demand: {heatwave['forecast_demand']:.1f} MLD (+{heatwave['demand_change_pct']:.1f}%)")
    print(f"     Stress ratio: {heatwave['stress_ratio']:.2f}x")
    print(f"     Risk: {heatwave['risk_category']} - {heatwave['risk_description']}")
    if heatwave['deficit_mld'] > 0:
        print(f"     ⚠️  Deficit: {heatwave['deficit_mld']:.1f} MLD")
    
    # Scenario 2: Rainfall deficit
    print("\n   ─── SCENARIO 2: Rainfall Deficit (40% below normal) ───")
    drought = simulator.apply_rainfall_change(rainfall_change_pct=-40)
    print(f"     Forecast demand: {drought['forecast_demand']:.1f} MLD (+{drought['demand_change_pct']:.1f}%)")
    print(f"     Stress ratio: {drought['stress_ratio']:.2f}x")
    print(f"     Risk: {drought['risk_category']} - {drought['risk_description']}")
    
    # Scenario 3: Population surge
    print("\n   ─── SCENARIO 3: Population Surge (8% increase) ───")
    pop_surge = simulator.apply_population_surge(growth_pct=8, duration_days=30)
    print(f"     Forecast demand: {pop_surge['forecast_demand']:.1f} MLD (+{pop_surge['demand_change_pct']:.1f}%)")
    print(f"     Stress ratio: {pop_surge['stress_ratio']:.2f}x")
    print(f"     Risk: {pop_surge['risk_category']}")
    
    # Scenario 4: Festival overlap
    print("\n   ─── SCENARIO 4: Festival Overlap (3 major festivals) ───")
    festival = simulator.apply_festival_overlap(num_festivals=3, avg_attendees=100000)
    print(f"     Forecast demand: {festival['forecast_demand']:.1f} MLD (+{festival['demand_change_pct']:.1f}%)")
    print(f"     Stress ratio: {festival['stress_ratio']:.2f}x")
    print(f"     Risk: {festival['risk_category']}")
    
    # Scenario 5: Industrial growth
    print("\n   ─── SCENARIO 5: Industrial Activity Surge (+25%) ───")
    industrial = simulator.apply_industrial_change(change_pct=25)
    print(f"     Forecast demand: {industrial['forecast_demand']:.1f} MLD (+{industrial['demand_change_pct']:.1f}%)")
    print(f"     Stress ratio: {industrial['stress_ratio']:.2f}x")
    print(f"     Risk: {industrial['risk_category']}")
    
    # Scenario 6: Combined stress
    print("\n   ─── SCENARIO 6: COMBINED CRISIS (Multiple events) ───")
    combined = simulator.combine_scenarios([heatwave, drought, festival])
    print(f"     Combined scenarios: Heatwave + Drought + Festival")
    print(f"     Forecast demand: {combined['forecast_demand']:.1f} MLD (+{combined['demand_change_pct']:.1f}%)")
    print(f"     Stress ratio: {combined['stress_ratio']:.2f}x")
    print(f"     Risk: {combined['risk_category']}")
    if combined['deficit_mld'] > 0:
        print(f"     🚨 CRITICAL DEFICIT: {combined['deficit_mld']:.1f} MLD")
    
    # Generate detailed report for worst scenario
    print("\n   ─── DETAILED ANALYSIS: COMBINED CRISIS SCENARIO ───")
    report = simulator.generate_scenario_report(combined)
    print(report)
    
    # Show first few recommendations
    print("\n   Top 5 Recommended Actions:")
    for i, rec in enumerate(combined['recommendations'][:5], 1):
        print(f"     {i}. {rec}")
    
    # Comparison of all scenarios
    print("\n   ─── SCENARIO COMPARISON ───")
    all_scenarios = [heatwave, drought, pop_surge, festival, industrial, combined]
    comparison = simulator.compare_scenarios(all_scenarios)
    print(comparison)
    
    print("\n" + "=" * 70)
    print("FORECASTING ENGINE READY FOR DEPLOYMENT")
    print("=" * 70)


class WaterDistributionRecommender:

    def __init__(self):

        self.zones = {}
        self.allocation_history = []
        self.shortage_alerts = []

        self.priority_levels = {
            'critical': 1,     
            'essential': 2,  
            'standard': 3,    
            'commercial': 4,   
            'industrial': 5,    
            'discretionary': 6  
        }
    
    def add_zone(self, zone_id: str, priority: str, min_demand_mld: float, 
                 max_demand_mld: float, current_population: int = None):

        self.zones[zone_id] = {
            'priority': priority,
            'priority_level': self.priority_levels.get(priority, 5),
            'min_demand': min_demand_mld,
            'max_demand': max_demand_mld,
            'population': current_population,
            'allocated_today': 0,
            'actual_demand': 0
        }
    
    def recommend_daily_release(self, 
                               forecasted_demand_mld: float,
                               reservoir_capacity_mld: float,
                               current_storage_mld: float,
                               max_daily_supply_mld: float,
                               safety_buffer_percentage: float = 10.0) -> Dict:

        safety_buffer_mld = (safety_buffer_percentage / 100) * reservoir_capacity_mld
        
        usable_storage = current_storage_mld - safety_buffer_mld
        usable_storage = max(usable_storage, 0)
        
        min_needs = self._calculate_minimum_needs()
        
        if usable_storage < 0:
            release = min(forecasted_demand_mld, max_daily_supply_mld)
            status = 'CRITICAL'
            reason = f"Storage {current_storage_mld:.1f} MLD below safety buffer {safety_buffer_mld:.1f} MLD"
        
        elif forecasted_demand_mld <= max_daily_supply_mld:
            release = min(forecasted_demand_mld, max_daily_supply_mld)
            storage_ratio = current_storage_mld / reservoir_capacity_mld
            if storage_ratio > 0.85:
                release = min(release, max_daily_supply_mld * 0.95)
            status = 'ADEQUATE'
            reason = f"Supply {release:.1f} MLD covers demand {forecasted_demand_mld:.1f} MLD"
        
        else:
            release = max_daily_supply_mld
            status = 'SHORTAGE'
            reason = f"Demand {forecasted_demand_mld:.1f} MLD exceeds max supply {max_daily_supply_mld:.1f} MLD"
        
        storage_after_release = current_storage_mld - release
        days_to_empty = storage_after_release / (release + 0.01) if release > 0 else float('inf')
        
        return {
            'recommended_release_mld': round(release, 2),
            'status': status,
            'reason': reason,
            'current_storage_mld': round(current_storage_mld, 2),
            'storage_after_release_mld': round(max(storage_after_release, 0), 2),
            'safety_buffer_mld': round(safety_buffer_mld, 2),
            'usable_storage_mld': round(usable_storage, 2),
            'forecasted_total_demand_mld': round(forecasted_demand_mld, 2),
            'max_supply_mld': max_daily_supply_mld,
            'minimum_critical_needs_mld': round(min_needs, 2),
            'days_to_empty_at_current_rate': round(days_to_empty, 1),
            'storage_health': self._assess_storage_health(
                current_storage_mld, reservoir_capacity_mld, safety_buffer_mld
            )
        }
    
    def allocate_to_zones(self, 
                         total_available_mld: float,
                         zone_demands: Dict[str, float],
                         apply_rationing: bool = False) -> Dict:

        allocation = {}
        sorted_zones = sorted(
            self.zones.items(),
            key=lambda x: x[1]['priority_level']
        )
        
        remaining_supply = total_available_mld
        total_demand = sum(zone_demands.values())
        shortage_ratio = remaining_supply / total_demand if total_demand > 0 else 1.0
        
        allocation_details = []
        
        for zone_id, zone_info in sorted_zones:
            if zone_info['priority_level'] <= self.priority_levels['critical']:
                min_needed = zone_info['min_demand']
                allocation[zone_id] = min_needed
                remaining_supply -= min_needed
                allocation_details.append({
                    'zone_id': zone_id,
                    'priority': zone_info['priority'],
                    'demand_mld': zone_demands.get(zone_id, min_needed),
                    'allocated_mld': min_needed,
                    'allocation_percentage': 100.0,
                    'rationing_status': 'NO RATIONING'
                })
        
        for zone_id, zone_info in sorted_zones:
            if (zone_info['priority_level'] > self.priority_levels['critical'] and
                zone_info['priority_level'] <= self.priority_levels['essential']):
                min_needed = zone_info['min_demand']
                allocation[zone_id] = min_needed
                remaining_supply -= min_needed
                allocation_details.append({
                    'zone_id': zone_id,
                    'priority': zone_info['priority'],
                    'demand_mld': zone_demands.get(zone_id, min_needed),
                    'allocated_mld': min_needed,
                    'allocation_percentage': 100.0,
                    'rationing_status': 'MINIMUM GUARANTEED'
                })
        
        for zone_id, zone_info in sorted_zones:
            if zone_id not in allocation:
                demand = zone_demands.get(zone_id, zone_info['max_demand'])
                
                if remaining_supply > 0:
                    allocated = min(demand * shortage_ratio, remaining_supply)
                    allocation[zone_id] = max(0, allocated)
                    remaining_supply -= allocation[zone_id]
                    percentage = (allocation[zone_id] / demand * 100) if demand > 0 else 0
                else:
                    allocation[zone_id] = 0
                    percentage = 0
                
                if percentage >= 95:
                    rationing_status = 'NO RATIONING'
                elif percentage >= 75:
                    rationing_status = '⚠️  MILD RATIONING (25% reduction)'
                elif percentage >= 50:
                    rationing_status = '⚠️  MODERATE RATIONING (50% reduction)'
                elif percentage > 0:
                    rationing_status = '🔴 SEVERE RATIONING (70%+ reduction)'
                else:
                    rationing_status = '🔴 COMPLETE CUTOFF'
                
                allocation_details.append({
                    'zone_id': zone_id,
                    'priority': zone_info['priority'],
                    'demand_mld': demand,
                    'allocated_mld': allocation[zone_id],
                    'allocation_percentage': round(percentage, 1),
                    'rationing_status': rationing_status
                })
        
        total_allocated = sum(allocation.values())
        total_shortage = max(0, total_demand - total_allocated)
        shortage_percentage = (total_shortage / total_demand * 100) if total_demand > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_available_mld': round(total_available_mld, 2),
            'total_demand_mld': round(total_demand, 2),
            'total_allocated_mld': round(total_allocated, 2),
            'total_shortage_mld': round(total_shortage, 2),
            'shortage_percentage': round(shortage_percentage, 1),
            'shortage_status': (
                '✅ NO SHORTAGE' if shortage_percentage <= 5 else
                '⚠️  MINOR SHORTAGE' if shortage_percentage <= 15 else
                '⚠️  MODERATE SHORTAGE' if shortage_percentage <= 30 else
                '🔴 SEVERE SHORTAGE'
            ),
            'zone_allocations': allocation_details,
            'allocation_by_zone': allocation
        }
    
    def calculate_rationing_schedule(self, shortage_percentage: float) -> Dict:

        if shortage_percentage <= 5:
            return {
                'status': 'NO RATIONING',
                'schedule': 'All hours available',
                'industrial_restriction': 'No restriction',
                'agricultural_restriction': 'No restriction',
                'commercial_restriction': 'No restriction'
            }
        
        elif shortage_percentage <= 15:
            return {
                'status': 'MILD RATIONING',
                'schedule': 'Reduced during 10 AM - 5 PM peak',
                'industrial_restriction': '20% reduction during peak hours',
                'agricultural_restriction': '30% reduction during peak hours',
                'commercial_restriction': 'Non-essential sprinklers prohibited 8 AM - 8 PM',
                'public_message': 'Avoid washing cars/outdoor watering during peak. Use water wisely.'
            }
        
        elif shortage_percentage <= 30:
            return {
                'status': 'MODERATE RATIONING',
                'schedule': 'Supply cuts 9 AM - 6 PM daily (3 hours in 3 areas at a time)',
                'industrial_restriction': '50% reduction all hours, no operation 2-4 PM',
                'agricultural_restriction': 'Only early morning (5-7 AM) irrigation allowed',
                'commercial_restriction': 'All non-essential outdoor water prohibited',
                'public_message': 'Supply cuts during peak. Store water. Cold showers recommended. Toilets use stored water.',
                'duration_estimate': 'Expected 7-14 days'
            }
        
        else: 
            return {
                'status': 'SEVERE RATIONING',
                'schedule': '4-8 hour supply cuts, 2-3 times weekly',
                'industrial_restriction': '70% reduction, many facilities closed',
                'agricultural_restriction': 'All irrigation prohibited',
                'commercial_restriction': 'All non-essential use prohibited',
                'public_message': 'Severe shortage. Use water only for drinking/cooking/sanitation. Expect 4-8 hour supply cuts.',
                'duration_estimate': 'Ongoing crisis - monitor daily updates',
                'emergency_measures': [
                    'Mobile water tanker distribution in deficient areas',
                    'Emergency desalination/recycled water activation',
                    'Community water storage points established'
                ]
            }
    
    def generate_allocation_report(self, allocation_result: Dict, 
                                  reservoir_info: Dict) -> str:

        report = []
        report.append("\n" + "=" * 80)
        report.append("WATER DISTRIBUTION ALLOCATION REPORT")
        report.append("=" * 80)
        report.append(f"\n📅 Generated: {allocation_result['timestamp']}")
        
        report.append(f"\n📊 OVERALL STATUS: {allocation_result['shortage_status']}")
        report.append(f"   Total Demand: {allocation_result['total_demand_mld']} MLD")
        report.append(f"   Available Supply: {allocation_result['total_available_mld']} MLD")
        report.append(f"   Total Allocation: {allocation_result['total_allocated_mld']} MLD")
        report.append(f"   Total Shortage: {allocation_result['total_shortage_mld']} MLD ({allocation_result['shortage_percentage']}%)")
        
        report.append("\n📍 ZONE-WISE ALLOCATION:")
        report.append("-" * 80)
        report.append(f"{'Zone':<20} {'Priority':<12} {'Demand':<12} {'Allocated':<12} {'%':<8} {'Status':<25}")
        report.append("-" * 80)
        
        for zone in allocation_result['zone_allocations']:
            zone_id = zone['zone_id'][:15] 
            priority = zone['priority'][:11]
            demand = f"{zone['demand_mld']:.1f}"
            allocated = f"{zone['allocated_mld']:.1f}"
            percentage = f"{zone['allocation_percentage']:.0f}%"
            status = zone['rationing_status'][:24]
            
            report.append(f"{zone_id:<20} {priority:<12} {demand:<12} {allocated:<12} {percentage:<8} {status:<25}")
        
        report.append("-" * 80)
        
        if allocation_result['shortage_percentage'] > 5:
            rationing = self.calculate_rationing_schedule(allocation_result['shortage_percentage'])
            report.append(f"\n⏰ RATIONING SCHEDULE: {rationing['status']}")
            report.append(f"   When: {rationing.get('schedule', 'N/A')}")
            report.append(f"   Industrial: {rationing.get('industrial_restriction', 'N/A')}")
            report.append(f"   Agriculture: {rationing.get('agricultural_restriction', 'N/A')}")
            report.append(f"   Commercial: {rationing.get('commercial_restriction', 'N/A')}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def _calculate_minimum_needs(self) -> float:
        min_needs = 0
        for zone in self.zones.values():
            if zone['priority_level'] <= self.priority_levels['essential']:
                min_needs += zone['min_demand']
        return min_needs
    
    def _assess_storage_health(self, current: float, capacity: float, 
                              buffer: float) -> str:
        ratio = current / capacity if capacity > 0 else 0
        
        if current < buffer:
            return '🔴 CRITICAL - Below safety buffer'
        elif ratio < 0.25:
            return '🔴 VERY LOW'
        elif ratio < 0.5:
            return '🟠 LOW'
        elif ratio < 0.75:
            return '🟡 MODERATE'
        elif ratio < 0.9:
            return '🟢 HEALTHY'
        else:
            return '💧 FULL'

class WaterRiskAlertManager:
    
    def __init__(self):
        self.alert_history = []
        self.stress_index_history = []
        
        self.thresholds = {
            'green': (0, 35),          
            'yellow': (35, 50),        
            'orange': (50, 75),        
            'red': (75, 100)           
        }

        self.alert_levels = {
            'green': {'emoji': '🟢', 'severity': 0, 'name': 'SAFE'},
            'yellow': {'emoji': '🟡', 'severity': 1, 'name': 'WATCH'},
            'orange': {'emoji': '🟠', 'severity': 2, 'name': 'PREPARE'},
            'red': {'emoji': '🔴', 'severity': 3, 'name': 'CRITICAL'}
        }
    
    def compute_water_stress_index(self,
                                   forecasted_demand_mld: float,
                                   current_storage_mld: float,
                                   reservoir_capacity_mld: float,
                                   max_daily_supply_mld: float,
                                   actual_inflow_mld: float = 0,
                                   trend_days: int = 7) -> Dict:
        available_supply = min(max_daily_supply_mld, current_storage_mld + actual_inflow_mld)
        if forecasted_demand_mld > 0:
            supply_ratio = available_supply / forecasted_demand_mld
        else:
            supply_ratio = 1.0
 
        supply_stress = max(0, min(100, (1.0 - supply_ratio) * 100))

        storage_ratio = current_storage_mld / reservoir_capacity_mld if reservoir_capacity_mld > 0 else 0

        if storage_ratio < 0.10:
            storage_stress = 90 + (10 * (0.10 - storage_ratio) / 0.10)  
        elif storage_ratio < 0.25:
            storage_stress = 75 + (15 * (0.25 - storage_ratio) / 0.15)  
        elif storage_ratio < 0.5:
            storage_stress = 40 + (35 * (0.5 - storage_ratio) / 0.25)   
        else:
            storage_stress = 40 * (1.0 - storage_ratio)                  
        
        daily_change = (actual_inflow_mld - forecasted_demand_mld)
        days_to_empty = current_storage_mld / (forecasted_demand_mld + 0.01) if forecasted_demand_mld > 0 else float('inf')
        
        if days_to_empty < 7:
            trend_stress = 80  
        elif days_to_empty < 14:
            trend_stress = 60  
        elif days_to_empty < 30:
            trend_stress = 40
        elif daily_change > 0:
            trend_stress = 10  
        else:
            trend_stress = 30  
        
        safety_buffer = 0.10 * reservoir_capacity_mld
        if current_storage_mld < safety_buffer:
            buffer_stress = 100  
        else:
            remaining_usable = current_storage_mld - safety_buffer
            buffer_stress = max(0, 50 - (remaining_usable / reservoir_capacity_mld) * 100)
        
        stress_index = (
            supply_stress * 0.40 +
            storage_stress * 0.30 +
            trend_stress * 0.20 +
            buffer_stress * 0.10
        )
        
        stress_index = max(0, min(100, stress_index))
        
        alert_level = self._get_alert_level(stress_index)
        
        return {
            'stress_index': round(stress_index, 2),
            'alert_level': alert_level,
            'component_breakdown': {
                'supply_stress': round(supply_stress, 2),
                'storage_stress': round(storage_stress, 2),
                'trend_stress': round(trend_stress, 2),
                'buffer_stress': round(buffer_stress, 2)
            },
            'metrics': {
                'supply_ratio': round(supply_ratio, 3),
                'storage_ratio': round(storage_ratio, 3),
                'days_to_empty': round(days_to_empty, 1),
                'daily_change_mld': round(daily_change, 2)
            }
        }
    
    def predict_shortages_forward(self,
                                  current_storage_mld: float,
                                  forecasted_demands: List[float],
                                  forecasted_inflows: List[float] = None,
                                  max_daily_supply_mld: float = 165.0) -> Dict:

        if forecasted_inflows is None:
            forecasted_inflows = [0.0] * len(forecasted_demands)
        
        storage = current_storage_mld
        daily_forecasts = []
        shortage_days = []
        
        for day in range(min(7, len(forecasted_demands))):
            demand = forecasted_demands[day]
            inflow = forecasted_inflows[day] if day < len(forecasted_inflows) else 0.0
            
            available = min(max_daily_supply_mld, storage + inflow)
            
            if demand > available:
                shortage = demand - available
                shortage_pct = (shortage / demand) * 100
                status = 'SHORTAGE'
            else:
                shortage = 0
                shortage_pct = 0
                status = 'ADEQUATE'
            
            storage = max(0, storage + inflow - available)
            
            alert = self._get_shortage_alert(shortage_pct, storage)
            
            daily_forecasts.append({
                'day': day + 1,
                'forecasted_demand': round(demand, 2),
                'expected_inflow': round(inflow, 2),
                'available_supply': round(available, 2),
                'shortage_mld': round(shortage, 2),
                'shortage_pct': round(shortage_pct, 1),
                'projected_storage_eod': round(storage, 2),
                'status': status,
                'alert': alert
            })
        
            if shortage > 0:
                shortage_days.append(day + 1)

        max_shortage = max([f['shortage_pct'] for f in daily_forecasts], default=0)
        shortage_risk = 'NONE'
        if max_shortage > 0 and max_shortage < 10:
            shortage_risk = 'LOW'
        elif max_shortage >= 10 and max_shortage < 25:
            shortage_risk = 'MODERATE'
        elif max_shortage >= 25:
            shortage_risk = 'HIGH'
        
        return {
            'forecast_window_days': 7,
            'current_storage': current_storage_mld,
            'shortage_risk_level': shortage_risk,
            'days_with_shortage': shortage_days,
            'max_shortage_pct': round(max_shortage, 1),
            'daily_forecasts': daily_forecasts,
            'summary': (
                f"Shortage expected on days: {shortage_days or 'None'} " +
                f"(max {round(max_shortage, 1)}% shortage)"
            )
        }
    
    def generate_alert_message(self,
                              stress_index: float,
                              alert_level: str,
                              forecasted_shortage_pct: float,
                              days_to_critical: int = None,
                              zone_impacts: Dict = None) -> Dict:

        alert_info = self.alert_levels.get(alert_level, self.alert_levels['yellow'])
        

        if alert_level == 'green':
            headline = "SAFE: Water supply adequate"
            description = f"Current stress index {stress_index:.0f}/100 indicates healthy water availability."
            public_message = "Water supply is sufficient. Continue normal usage."
            actions = [
                "Monitor daily stress index",
                "Maintain regular monitoring schedule",
                "Update public dashboard"
            ]
            duration = "Ongoing"
        
        elif alert_level == 'yellow':
            headline = "WATCH: Monitor water situation closely"
            description = f"Stress index {stress_index:.0f}/100 shows potential issues ahead."
            public_message = "Please conserve water. Reduce non-essential use."
            actions = [
                "Increase monitoring frequency (every 6 hours)",
                "Prepare contingency plans",
                "Alert stakeholders to watch alert",
                "Request voluntary conservation from public",
                "Check equipment and backup systems"
            ]
            duration = "24-72 hours (or until conditions improve)"
        
        elif alert_level == 'orange':
            headline = "PREPARE: Shortages likely within 3-7 days"
            description = f"Stress index {stress_index:.0f}/100 indicates significant shortage risk."
            public_message = "Rationing may be required. Prepare now."
            actions = [
                "Activate emergency response team",
                "Implement mandatory conservation measures",
                "Begin rationing schedule",
                "Deploy mobile water tankers to deficient zones",
                "Activate recycled water facilities (if available)",
                "Issue official shortage notice to all sectors",
                "Monitor hourly and adjust release strategy"
            ]
            duration = "3-7 days expected"
        
        else:  # red
            headline = "CRITICAL: Emergency water shortage"
            description = f"Stress index {stress_index:.0f}/100 - Critical shortage in progress."
            public_message = "EMERGENCY: Use water only for essential needs (drinking, cooking, sanitation)."
            actions = [
                "DECLARE WATER EMERGENCY immediately",
                "Implement severe rationing (4-8 hour daily cutoffs)",
                "Deploy all mobile water tanker resources",
                "Activate desalination at maximum capacity",
                "Implement voluntary industry shutdowns",
                "Close non-essential public water use (fountains, etc.)",
                "Coordinate with neighboring utilities for emergency supply",
                "Brief emergency operations center hourly",
                "Alert hospitals and critical services of status"
            ]
            duration = "Ongoing crisis - continuous monitoring"
        
  
        time_to_critical_text = ""
        if days_to_critical is not None and days_to_critical > 0:
            if days_to_critical == 1:
                time_to_critical_text = "Crisis expected within 24 hours."
            elif days_to_critical <= 3:
                time_to_critical_text = f"Crisis expected in {days_to_critical} days."
            else:
                time_to_critical_text = f"Crisis possible in {days_to_critical} days if trend continues."
   
        zone_impact_text = ""
        if zone_impacts:
            affected_zones = [z for z, impact in zone_impacts.items() if impact > 0]
            if affected_zones:
                zone_impact_text = f"\nZones affected: {', '.join(affected_zones)}"
        
        return {
            'timestamp': datetime.now().isoformat(),
            'alert_level': alert_level,
            'severity_number': alert_info['severity'],  # 0=green, 1=yellow, 2=orange, 3=red
            'emoji': alert_info['emoji'],
            'name': alert_info['name'],
            'stress_index': round(stress_index, 2),
            'headline': f"{alert_info['emoji']} {headline}",
            'description': description,
            'public_message': public_message,
            'time_to_critical': time_to_critical_text,
            'zone_impacts': zone_impact_text,
            'recommended_actions': actions,
            'duration': duration,
            'forecasted_shortage_pct': round(forecasted_shortage_pct, 1),
            'alert_text': (
                f"\n{alert_info['emoji']} {headline}\n" +
                f"Stress Index: {stress_index:.0f}/100\n" +
                f"Description: {description}\n" +
                f"{time_to_critical_text}\n" +
                f"Expected Duration: {duration}\n" +
                f"Public Message: {public_message}\n" +
                f"\nRecommended Actions:\n" +
                "\n".join([f"  {i+1}. {action}" for i, action in enumerate(actions)])
            )
        }
    
    def generate_comprehensive_alert_report(self,
                                           stress_index: float,
                                           shortage_forecast: Dict,
                                           zone_allocations: Dict = None) -> str:

        
        alert_level = self._get_alert_level(stress_index)
        alert_info = self.alert_levels[alert_level]
        
        report = []
        report.append("\n" + "=" * 80)
        report.append("WATER RISK AND ALERT REPORT")
        report.append("=" * 80)
        report.append(f"\n📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Current Status
        report.append(f"\n🚨 CURRENT ALERT STATUS: {alert_info['emoji']} {alert_info['name']}")
        report.append(f"   Stress Index: {stress_index:.1f}/100")
        
        # Stress Category
        if stress_index < 35:
            report.append("   Status: Safe, ample supply")
        elif stress_index < 50:
            report.append("   Status: Monitor closely")
        elif stress_index < 75:
            report.append("   Status: Prepare for restrictions")
        else:
            report.append("   Status: CRITICAL - Emergency conditions")
        
   
        report.append(f"\n📊 7-DAY SHORTAGE FORECAST:")
        report.append(f"   Overall Risk: {shortage_forecast['shortage_risk_level']}")
        report.append(f"   Max Expected Shortage: {shortage_forecast['max_shortage_pct']}%")
        if shortage_forecast['days_with_shortage']:
            report.append(f"   Shortage Days: {shortage_forecast['days_with_shortage']}")

        report.append(f"\n   Daily Breakdown:")
        report.append(f"   {'Day':<5} {'Demand':<10} {'Supply':<10} {'Shortage':<12} {'Status':<12}")
        report.append(f"   {'-'*49}")
        
        for day_forecast in shortage_forecast['daily_forecasts']:
            day = day_forecast['day']
            demand = f"{day_forecast['forecasted_demand']:.1f}"
            supply = f"{day_forecast['available_supply']:.1f}"
            shortage = f"{day_forecast['shortage_pct']:.1f}%"
            status = day_forecast['status']
            report.append(f"   Day {day:<1} {demand:<10} {supply:<10} {shortage:<12} {status:<12}")
        
        if zone_allocations:
            report.append(f"\n📍 ZONE IMPACTS:")
            for zone in zone_allocations.get('zone_allocations', []):
                if zone['allocation_percentage'] < 100:
                    report.append(f"   {zone['zone_id']}: {zone['allocation_percentage']:.0f}% allocation")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def _get_alert_level(self, stress_index: float) -> str:

        if stress_index < self.thresholds['yellow'][0]:
            return 'green'
        elif stress_index < self.thresholds['orange'][0]:
            return 'yellow'
        elif stress_index < self.thresholds['red'][0]:
            return 'orange'
        else:
            return 'red'
    
    def _get_shortage_alert(self, shortage_pct: float, storage_remaining: float) -> str:
        if shortage_pct == 0:
            return 'SAFE'
        elif shortage_pct < 10:
            return 'MILD'
        elif shortage_pct < 25:
            return 'MODERATE'
        else:
            return 'SEVERE'

class WaterRecyclingClassifier:

    def __init__(self):

        self.categories = {
            'greywater': {
                'name': 'Greywater',
                'source': 'Sinks, showers, washing machines, baths',
                'recyclability': 'Yes',
                'recyclability_score': 0.85,  # 0-1 scale
                'treatment_complexity': 'Low-Medium',
                'typical_reuse_purposes': [
                    'Toilet flushing',
                    'Landscape irrigation',
                    'Industrial cooling',
                    'Process water',
                    'Groundwater recharge'
                ],
                'recoverable_percentage': 0.65,  
                'freshwater_demand_reduction': 0.30,  
                'quality_parameters': {
                    'suspended_solids': 'High',
                    'organic_content': 'Medium',
                    'pathogens': 'Low-Medium',
                    'chemicals': 'Medium (soaps, detergents)'
                },
                'estimated_volume_pld': 50,  # Per liter per day per person
                'cost_per_kld': 8,  # Cost to recycle (INR per kiloliter)
                'color': '#4A90E2',  # Blue
                'emoji': '🚿'
            },
            'blackwater': {
                'name': 'Blackwater',
                'source': 'Human waste and sewage water',
                'recyclability': 'Partial',
                'recyclability_score': 0.40,  # Lower - requires intensive treatment
                'treatment_complexity': 'High',
                'typical_reuse_purposes': [
                    'Biogas generation',
                    'Fertilizer production (after composting)',
                    'Industrial reuse (limited)',
                    'Non-potable irrigation (after advanced treatment)'
                ],
                'recoverable_percentage': 0.35,  # 35% recovery after treatment
                'freshwater_demand_reduction': 0.15,  # 15% reduction potential
                'quality_parameters': {
                    'suspended_solids': 'Very High',
                    'organic_content': 'Very High',
                    'pathogens': 'High',
                    'chemicals': 'High (medications, cleaning products)'
                },
                'estimated_volume_pld': 45,  # Per liter per day per person
                'cost_per_kld': 18,  # Higher cost to recycle
                'color': '#E74C3C',  # Red
                'emoji': '🚽'
            },
            'industrial_wastewater': {
                'name': 'Industrial Wastewater',
                'source': 'Manufacturing, processing, chemical plants',
                'recyclability': 'Partial',
                'recyclability_score': 0.60,  # Variable by industry
                'treatment_complexity': 'High',
                'typical_reuse_purposes': [
                    'Cooling tower makeup',
                    'Boiler feed (after treatment)',
                    'Process water (same industry)',
                    'Dust suppression',
                    'General washing'
                ],
                'recoverable_percentage': 0.55,  # 55% recovery potential
                'freshwater_demand_reduction': 0.45,  # 45% reduction potential (high volume)
                'quality_parameters': {
                    'suspended_solids': 'High',
                    'organic_content': 'Medium-High',
                    'pathogens': 'Low',
                    'chemicals': 'Very High (heavy metals, oils, solvents)'
                },
                'estimated_volume_pld': 200,  # Much higher per unit (plants vary)
                'cost_per_kld': 12,  # Medium cost to recycle
                'color': '#F39C12',  # Orange
                'emoji': '🏭'
            },
            'rainwater': {
                'name': 'Rainwater Harvesting',
                'source': 'Precipitation, roof catchment',
                'recyclability': 'Yes',
                'recyclability_score': 0.90,  # Highest recyclability
                'treatment_complexity': 'Low',
                'typical_reuse_purposes': [
                    'Drinking water (after minimal treatment)',
                    'Landscape irrigation',
                    'Toilet flushing',
                    'Vehicle washing',
                    'Groundwater recharge (aquifer storage)'
                ],
                'recoverable_percentage': 0.80,  # 80% can be captured
                'freshwater_demand_reduction': 0.20,  # 20% reduction (seasonal)
                'quality_parameters': {
                    'suspended_solids': 'Low',
                    'organic_content': 'Low',
                    'pathogens': 'Low',
                    'chemicals': 'Low (dust, pollen)'
                },
                'estimated_volume_pld': 40,  # Per mm of rainfall per sqm
                'cost_per_kld': 3,  # Lowest cost to treat
                'color': '#27AE60',  # Green
                'emoji': '🌧️'
            },
            'treated_sewage': {
                'name': 'Treated Sewage Water',
                'source': 'Processed blackwater from treatment plants',
                'recyclability': 'Yes',
                'recyclability_score': 0.75,  # Good - already treated
                'treatment_complexity': 'Medium',
                'typical_reuse_purposes': [
                    'Non-potable irrigation',
                    'Landscape watering',
                    'Industrial process water',
                    'Toilet flushing',
                    'Dust suppression',
                    'Aquifer recharge'
                ],
                'recoverable_percentage': 0.90,  # 90% available (already collected)
                'freshwater_demand_reduction': 0.35,  # 35% reduction potential
                'quality_parameters': {
                    'suspended_solids': 'Low-Medium',
                    'organic_content': 'Low',
                    'pathogens': 'Low (post-treatment)',
                    'chemicals': 'Low (residual)'
                },
                'estimated_volume_pld': 120,  # From sewage treatment plant
                'cost_per_kld': 6,  # Lower cost (already treated)
                'color': '#9B59B6',  # Purple
                'emoji': '♻️'
            }
        }
    
    def get_category_info(self, category_id: str) -> Dict:
        """
        Get detailed information about a water category.
        
        Args:
            category_id: 'greywater', 'blackwater', 'industrial_wastewater', 
                        'rainwater', 'treated_sewage'
        
        Returns:
            Dictionary with complete category information
        """
        if category_id not in self.categories:
            return {'error': f'Category {category_id} not found'}
        
        cat = self.categories[category_id]
        
        return {
            'category_id': category_id,
            'name': cat['name'],
            'emoji': cat['emoji'],
            'source': cat['source'],
            'recyclability': cat['recyclability'],
            'recyclability_score': cat['recyclability_score'],
            'treatment_complexity': cat['treatment_complexity'],
            'reuse_purposes': cat['typical_reuse_purposes'],
            'recoverable_percentage': round(cat['recoverable_percentage'] * 100, 1),
            'freshwater_reduction_potential': round(cat['freshwater_demand_reduction'] * 100, 1),
            'quality_parameters': cat['quality_parameters'],
            'cost_per_kld': cat['cost_per_kld'],
            'estimated_daily_volume_pld': cat['estimated_volume_pld']
        }
    
    def get_all_categories(self) -> List[Dict]:
        """Get summary of all water categories."""
        summary = []
        
        for cat_id in self.categories:
            cat = self.categories[cat_id]
            summary.append({
                'id': cat_id,
                'name': cat['name'],
                'emoji': cat['emoji'],
                'recyclability': cat['recyclability'],
                'recyclability_score': round(cat['recyclability_score'] * 100, 1),
                'treatment_difficulty': cat['treatment_complexity'],
                'freshwater_reduction': round(cat['freshwater_demand_reduction'] * 100, 1),
                'color': cat['color']
            })
        
        return summary
    
    def calculate_recycling_potential(self,
                                      greywater_volume: float = 0,
                                      blackwater_volume: float = 0,
                                      industrial_volume: float = 0,
                                      rainwater_volume: float = 0,
                                      treated_sewage_volume: float = 0) -> Dict:
        """
        Calculate total recycling potential for mixed water sources.
        
        Args:
            greywater_volume: Greywater available (MLD)
            blackwater_volume: Blackwater available (MLD)
            industrial_volume: Industrial wastewater available (MLD)
            rainwater_volume: Rainwater collected (MLD)
            treated_sewage_volume: Treated sewage available (MLD)
        
        Returns:
            Dictionary with recycling potential analysis
        """
        
        volumes = {
            'greywater': greywater_volume,
            'blackwater': blackwater_volume,
            'industrial_wastewater': industrial_volume,
            'rainwater': rainwater_volume,
            'treated_sewage': treated_sewage_volume
        }
        
        total_input = sum(volumes.values())
        
        if total_input == 0:
            return {'error': 'No water volumes provided'}
        
        # Calculate recoverable for each source
        recoverable = {}
        freshwater_reduction = {}
        treatment_cost = {}
        
        for source_id, volume in volumes.items():
            if volume > 0:
                cat = self.categories[source_id]
                
                # Recoverable volume
                recoverable[source_id] = volume * cat['recoverable_percentage']
                
                # Freshwater reduction contribution
                freshwater_reduction[source_id] = volume * cat['freshwater_demand_reduction']
                
                # Treatment cost
                treatment_cost[source_id] = (volume * 1000) * cat['cost_per_kld']  # Convert MLD to KLD
        
        # Total calculations
        total_recoverable = sum(recoverable.values())
        total_recovery_rate = (total_recoverable / total_input * 100) if total_input > 0 else 0
        
        total_freshwater_reduction = sum(freshwater_reduction.values())
        total_treatment_cost = sum(treatment_cost.values())
        
        # Recyclability weighted score
        weighted_score = 0
        for source_id, volume in volumes.items():
            if volume > 0 and total_input > 0:
                weight = volume / total_input
                recyclability = self.categories[source_id]['recyclability_score']
                weighted_score += weight * recyclability
        
        return {
            'total_input_volume_mld': round(total_input, 2),
            'total_recoverable_mld': round(total_recoverable, 2),
            'overall_recovery_rate': round(total_recovery_rate, 1),
            'overall_recyclability_score': round(weighted_score * 100, 1),
            'by_source': {
                source_id: {
                    'input_mld': round(volume, 2),
                    'recoverable_mld': round(recoverable.get(source_id, 0), 2),
                    'recyclability': self.categories[source_id]['recyclability'],
                    'freshwater_reduction_mld': round(freshwater_reduction.get(source_id, 0), 2)
                }
                for source_id, volume in volumes.items() if volume > 0
            },
            'total_freshwater_demand_reduction_mld': round(total_freshwater_reduction, 2),
            'estimated_treatment_cost_inr': round(total_treatment_cost, 2),
            'cost_per_mld_treated': round(total_treatment_cost / total_input, 2) if total_input > 0 else 0
        }
    
    def generate_recycling_recommendation(self, water_sources: Dict) -> Dict:
        """
        Generate prioritized recommendations for water recycling strategy.
        
        Args:
            water_sources: Dict with volumes of each water source
        
        Returns:
            Prioritized list of recycling recommendations
        """
        
        recommendations = []
        
        # Rank sources by recyclability score and volume potential
        ranked_sources = []
        
        for source_id, volume in water_sources.items():
            if volume > 0 and source_id in self.categories:
                cat = self.categories[source_id]
                
                # Score based on recyclability and volume
                score = (cat['recyclability_score'] * 0.6 + 
                        (volume / (sum(water_sources.values()) or 1)) * 0.4)
                
                ranked_sources.append({
                    'source_id': source_id,
                    'name': cat['name'],
                    'volume': volume,
                    'priority_score': score,
                    'recyclability': cat['recyclability'],
                    'cost_per_kld': cat['cost_per_kld']
                })
        
        # Sort by priority
        ranked_sources.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Generate recommendations
        for idx, source in enumerate(ranked_sources, 1):
            cat = self.categories[source['source_id']]
            
            recommendation = {
                'priority': idx,
                'source': f"{source['name']} {cat['emoji']}",
                'volume_available_mld': round(source['volume'], 2),
                'recyclability': source['recyclability'],
                'action': self._get_action_for_source(source['source_id']),
                'expected_recovery_mld': round(source['volume'] * cat['recoverable_percentage'], 2),
                'freshwater_reduction_potential': round(source['volume'] * cat['freshwater_demand_reduction'], 2),
                'estimated_cost_inr': round(source['volume'] * 1000 * source['cost_per_kld'], 2),
                'timeline': self._get_implementation_timeline(source['source_id'])
            }
            
            recommendations.append(recommendation)
        
        return {
            'recommendations': recommendations,
            'total_volume_to_recycle_mld': round(sum(water_sources.values()), 2),
            'summary': self._generate_recycling_summary(recommendations)
        }
    
    def calculate_demand_offset(self,
                               total_freshwater_demand: float,
                               recycled_water_available: float) -> Dict:
        """
        Calculate how recycled water reduces freshwater demand.
        
        Args:
            total_freshwater_demand: Total freshwater needed (MLD)
            recycled_water_available: Water available from recycling (MLD)
        
        Returns:
            Demand offset analysis
        """
        
        freshwater_after_recycling = max(0, total_freshwater_demand - recycled_water_available)
        demand_offset_pct = (recycled_water_available / total_freshwater_demand * 100) if total_freshwater_demand > 0 else 0
        
        # Environmental impact
        water_saved_million_liters = recycled_water_available * 1000  # Convert MLD to Million liters
        carbon_saved_tons = water_saved_million_liters * 0.5  # Rough estimate: 0.5 ton CO2 per million liters
        energy_saved_kwh = recycled_water_available * 2.5 * 365  # kWh per day × 365 days
        
        # Cost comparison
        cost_freshwater_per_mld = 25  # Typical cost in INR (varies by city)
        cost_recycled_per_mld = 8    # Typical cost in INR
        
        freshwater_cost = total_freshwater_demand * cost_freshwater_per_mld
        recycled_cost = recycled_water_available * cost_recycled_per_mld
        freshwater_cost_after = freshwater_after_recycling * cost_freshwater_per_mld
        
        total_cost = recycled_cost + freshwater_cost_after
        cost_savings = freshwater_cost - total_cost
        
        return {
            'freshwater_demand_original_mld': round(total_freshwater_demand, 2),
            'recycled_water_used_mld': round(recycled_water_available, 2),
            'freshwater_demand_after_recycling_mld': round(freshwater_after_recycling, 2),
            'demand_offset_percentage': round(demand_offset_pct, 1),
            'environmental_impact': {
                'water_saved_million_liters': round(water_saved_million_liters, 2),
                'co2_emissions_avoided_tons': round(carbon_saved_tons, 2),
                'energy_saved_kwh': round(energy_saved_kwh, 2)
            },
            'financial_analysis': {
                'original_freshwater_cost_inr': round(freshwater_cost, 2),
                'recycled_water_total_cost_inr': round(total_cost, 2),
                'annual_cost_savings_inr': round(cost_savings * 365, 2),
                'cost_reduction_percentage': round((cost_savings / freshwater_cost * 100) if freshwater_cost > 0 else 0, 1)
            }
        }
    
    def _get_action_for_source(self, source_id: str) -> str:
        """Get recommended action for a specific water source."""
        actions = {
            'greywater': 'Install greywater recycling systems in buildings (showers, sinks)',
            'blackwater': 'Connect to sewage treatment plant with advanced processing',
            'industrial_wastewater': 'Install on-site treatment and closed-loop recycling systems',
            'rainwater': 'Install rainwater harvesting and storage infrastructure',
            'treated_sewage': 'Integrate treated sewage into irrigation and non-potable uses'
        }
        return actions.get(source_id, 'Implement recycling system')
    
    def _get_implementation_timeline(self, source_id: str) -> str:
        """Get typical implementation timeline."""
        timelines = {
            'rainwater': '3-6 months',
            'greywater': '6-12 months',
            'treated_sewage': '12-18 months',
            'industrial_wastewater': '12-24 months',
            'blackwater': '18-36 months'
        }
        return timelines.get(source_id, '12 months')
    
    def _generate_recycling_summary(self, recommendations: List[Dict]) -> str:
        """Generate text summary of recycling strategy."""
        total_volume = sum(r['volume_available_mld'] for r in recommendations)
        total_recovery = sum(r['expected_recovery_mld'] for r in recommendations)
        total_freshwater_reduction = sum(r['freshwater_reduction_potential'] for r in recommendations)
        
        summary = (
            f"Total recyclable water: {total_volume:.1f} MLD | "
            f"Expected recovery: {total_recovery:.1f} MLD | "
            f"Freshwater reduction: {total_freshwater_reduction:.1f} MLD | "
            f"Top priority: {recommendations[0]['source'] if recommendations else 'None'}"
        )
        
        return summary


if __name__ == '__main__':
    example_training_pipeline()
