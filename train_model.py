import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pickle
from datetime import datetime
import warnings
import calendar
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CovidMeteoPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def load_covid_data(self, file_path):
        """Charge les données COVID-19 et filtre pour la France"""
        print("Chargement des données COVID-19...")
        df = pd.read_csv(file_path)
        
        france_data = df[df['location'] == 'France'].copy()
        
        france_data['date'] = pd.to_datetime(france_data['date'])
        
        covid_features = ['date', 'new_cases', 'new_deaths', 'total_cases', 'total_deaths']
        france_data = france_data[covid_features]
        
        france_data = france_data.fillna(0)
        
        print(f"Données COVID-19 chargées: {len(france_data)} lignes")
        return france_data
    
    def load_meteo_data(self, file_path):
        """Charge les données météorologiques de toutes les villes et les étend au niveau quotidien"""
        print("Chargement des données météorologiques...")
        df = pd.read_csv(file_path, sep=';')
        
        meteo_data = df.copy()
        
        meteo_data['AAAAMM_str'] = meteo_data['AAAAMM'].astype(str)
        meteo_data['year'] = meteo_data['AAAAMM_str'].str[:4].astype(int)
        meteo_data['month'] = meteo_data['AAAAMM_str'].str[4:6].astype(int)
        meteo_data['date_month'] = pd.to_datetime(meteo_data[['year', 'month']].assign(day=1))
        
        meteo_data = meteo_data.rename(columns={
            'RR': 'precipitation_mm',
            'RR_ME': 'precipitation_median',
            'UMM': 'humidite_moyenne',
            'INST': 'insolation_hours',
            'TM': 'temperature_moyenne'
        })
        
        meteo_data = meteo_data.replace('', np.nan)
        numeric_cols = ['precipitation_mm', 'humidite_moyenne', 'insolation_hours', 'temperature_moyenne']
        meteo_data[numeric_cols] = meteo_data[numeric_cols].astype(float)
        meteo_data = meteo_data.fillna(meteo_data[numeric_cols].mean())
        
        print("Extension des données mensuelles au niveau quotidien...")
        daily_data = []
        
        for idx, (_, row) in enumerate(meteo_data.iterrows()):
            if idx % 100 == 0:
                print(f"  Progression: {idx}/{len(meteo_data)} lignes traitées")
                
            year, month = row['year'], row['month']
            
            days_in_month = calendar.monthrange(year, month)[1]
            
            for day in range(1, days_in_month + 1):
                daily_row = row.copy()
                daily_row['date'] = pd.Timestamp(year, month, day)
                daily_data.append(daily_row)
        
        daily_meteo = pd.DataFrame(daily_data)
        
        meteo_features = ['date', 'precipitation_mm', 'humidite_moyenne', 'insolation_hours', 'temperature_moyenne']
        daily_meteo = daily_meteo[meteo_features]
        
        print(f"Données météorologiques chargées: {len(meteo_data)} lignes mensuelles → {len(daily_meteo)} lignes quotidiennes")
        return daily_meteo
    
    def merge_data(self, covid_data, meteo_data):
        """Fusionne les données COVID et météo par date exacte (quotidienne)"""
        print("Fusion des données par date quotidienne...")
        
        covid_data['date'] = pd.to_datetime(covid_data['date'])
        meteo_data['date'] = pd.to_datetime(meteo_data['date'])
        
        merged_data = pd.merge(covid_data, meteo_data, on='date', how='inner')
        
        merged_data = merged_data.sort_values('date')
        
        print(f"Données fusionnées: {len(merged_data)} lignes quotidiennes")
        print(f"Période couverte: {merged_data['date'].min()} à {merged_data['date'].max()}")
        return merged_data
    
    def create_features(self, data):
        """Crée des features supplémentaires"""
        print("Création de features supplémentaires...")
        
        data['month'] = data['date'].dt.month
        data['season'] = data['date'].dt.month % 12 // 3 + 1
        
        data['new_cases_7d_avg'] = data['new_cases'].rolling(window=7, min_periods=1).mean()
        data['new_deaths_7d_avg'] = data['new_deaths'].rolling(window=7, min_periods=1).mean()
        
        data['temp_humidity_interaction'] = data['temperature_moyenne'] * data['humidite_moyenne']
        data['precipitation_binary'] = (data['precipitation_mm'] > 50).astype(int)
        
        data['new_cases_lag_1'] = data['new_cases'].shift(1)
        data['new_cases_lag_7'] = data['new_cases'].shift(7)
        
        # change les NaN pour avoir des 0 bien tranquilles
        data = data.fillna(0)
        
        return data
    
    def prepare_data(self, data, target='new_cases'):
        """Prépare les données pour l'entraînement"""
        print("Préparation des données pour l'entraînement...")
        
        feature_columns = [
            'precipitation_mm', 'humidite_moyenne', 'insolation_hours', 'temperature_moyenne',
            'month', 'season', 'new_cases_7d_avg', 'new_deaths_7d_avg',
            'temp_humidity_interaction', 'precipitation_binary',
            'new_cases_lag_1', 'new_cases_lag_7'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        
        X = data[available_features]
        y = data[target]
        
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        print(f"Features utilisées: {available_features}")
        print(f"Données finales: {len(X)} échantillons")
        
        return X, y, available_features
    
    def train_models(self, X, y):
        """Entraîne plusieurs modèles"""
        print("Entraînement des modèles...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Entraînement du modèle: {name}")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.3f}")
        
        self.models = results
        return results
    
    def plot_results(self, results):
        """Affiche les résultats et visualisations"""
        print("Génération des visualisations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Prédiction COVID-19 quotidienne basée sur les données météorologiques - France', fontsize=16)
        
        model_names = list(results.keys())
        rmse_scores = [results[name]['rmse'] for name in model_names]
        r2_scores = [results[name]['r2'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, rmse_scores, width, label='RMSE', alpha=0.8)
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('Comparaison RMSE des modèles')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        
        axes[0, 1].bar(x, r2_scores, alpha=0.8, color='orange')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].set_title('Comparaison R² des modèles')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_result = results[best_model_name]
        
        axes[1, 0].scatter(best_result['y_test'], best_result['y_pred'], alpha=0.6)
        axes[1, 0].plot([best_result['y_test'].min(), best_result['y_test'].max()], 
                       [best_result['y_test'].min(), best_result['y_test'].max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Valeurs réelles')
        axes[1, 0].set_ylabel('Prédictions')
        axes[1, 0].set_title(f'Prédictions vs Réalité - {best_model_name}')
        
        if 'Random Forest' in results:
            rf_model = results['Random Forest']['model']
            feature_names = [f'Feature_{i}' for i in range(len(rf_model.feature_importances_))]
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1, 1].set_yticks(range(len(feature_importance)))
            axes[1, 1].set_yticklabels(feature_importance['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Importance des features (Random Forest)')
        
        plt.tight_layout()
        plt.savefig('covid_meteo_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results):
        """Génère un rapport détaillé"""
        print("\n" + "="*60)
        print("RAPPORT D'ANALYSE - PRÉDICTION COVID-19 QUOTIDIENNE / MÉTÉO")
        print("="*60)
        
        print(f"\nDonnées utilisées:")
        print(f"- Période: Données COVID-19 et météorologiques fusionnées quotidiennement")
        print(f"- Localisation: France (données météo de toutes les villes)")
        print(f"- Variables météo: Précipitations, humidité, insolation, température")
        print(f"- Résolution temporelle: Prédictions jour par jour")
        
        print(f"\nRésultats des modèles:")
        print("-" * 40)
        
        best_model = max(results.keys(), key=lambda x: results[x]['r2'])
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  RMSE: {result['rmse']:.2f}")
            print(f"  MAE: {result['mae']:.2f}")
            print(f"  R²: {result['r2']:.3f}")
            
            if name == best_model:
                print(f"  ⭐ MEILLEUR MODÈLE")
        
        print(f"\nInterprétation:")
        print(f"- Le modèle {best_model} obtient les meilleures performances")
        print(f"- R² = {results[best_model]['r2']:.3f} indique la qualité de prédiction")
        print(f"- Les variables météorologiques contribuent à la prédiction des cas COVID")
        
        return best_model
    
    def save_best_model(self, results, feature_names):
        """Sauvegarde le meilleur modèle avec ses métadonnées"""
        print("Sauvegarde du meilleur modèle...")
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_result = results[best_model_name]
        best_model = best_result['model']
        
        model_package = {
            'model': best_model,
            'scaler': self.scaler,
            'model_name': best_model_name,
            'feature_names': feature_names,
            'performance_metrics': {
                'rmse': best_result['rmse'],
                'mae': best_result['mae'],
                'r2': best_result['r2'],
                'mse': best_result['mse']
            },
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': type(best_model).__name__
        }
        
        model_filename = f'best_covid_meteo_model_{best_model_name.replace(" ", "_").lower()}.pkl'
        joblib.dump(model_package, model_filename)
        
        metadata_filename = f'model_metadata_{best_model_name.replace(" ", "_").lower()}.txt'
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            f.write("MÉTADONNÉES DU MODÈLE COVID-19 / MÉTÉO\n")
            f.write("="*50 + "\n\n")
            f.write(f"Nom du modèle: {best_model_name}\n")
            f.write(f"Type de modèle: {type(best_model).__name__}\n")
            f.write(f"Date d'entraînement: {model_package['training_date']}\n\n")
            f.write("PERFORMANCES:\n")
            f.write(f"- RMSE: {best_result['rmse']:.2f}\n")
            f.write(f"- MAE: {best_result['mae']:.2f}\n")
            f.write(f"- R²: {best_result['r2']:.3f}\n")
            f.write(f"- MSE: {best_result['mse']:.2f}\n\n")
            f.write("FEATURES UTILISÉES:\n")
            for i, feature in enumerate(feature_names, 1):
                f.write(f"{i}. {feature}\n")
            f.write(f"\nFichier modèle: {model_filename}\n")
        
        print(f"Modèle sauvegardé: {model_filename}")
        print(f"Métadonnées sauvegardées: {metadata_filename}")
        
        return model_filename, metadata_filename
    
    def load_saved_model(self, model_filename):
        """Charge un modèle précédemment sauvegardé"""
        print(f"Chargement du modèle: {model_filename}")
        
        try:
            model_package = joblib.load(model_filename)
            
            print(f"Modèle chargé: {model_package['model_name']}")
            print(f"Type: {model_package['model_type']}")
            print(f"Entraîné le: {model_package['training_date']}")
            print(f"Performance R²: {model_package['performance_metrics']['r2']:.3f}")
            
            return model_package
            
        except FileNotFoundError:
            print(f"Erreur: Le fichier {model_filename} n'existe pas.")
            return None
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None
    
    def predict_with_saved_model(self, model_package, new_data):
        """Fait des prédictions avec le modèle sauvegardé"""
        if model_package is None:
            print("Aucun modèle chargé.")
            return None
        
        model = model_package['model']
        scaler = model_package['scaler']
        feature_names = model_package['feature_names']
        
        if not all(feature in new_data.columns for feature in feature_names):
            missing_features = [f for f in feature_names if f not in new_data.columns]
            print(f"Erreur: Features manquantes: {missing_features}")
            return None
        
        X = new_data[feature_names]
        
        if model_package['model_type'] == 'LinearRegression':
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def run_analysis(self, covid_file, meteo_file, save_model=True):
        """Exécute l'analyse complète"""
        print("Début de l'analyse COVID-19 / Météo")
        print("="*50)
        
        covid_data = self.load_covid_data(covid_file)
        meteo_data = self.load_meteo_data(meteo_file)
        
        merged_data = self.merge_data(covid_data, meteo_data)
        
        data_with_features = self.create_features(merged_data)
        
        X, y, feature_names = self.prepare_data(data_with_features)
        
        results = self.train_models(X, y)
        
        self.plot_results(results)
        
        best_model = self.generate_report(results)
        
        if save_model:
            model_file, metadata_file = self.save_best_model(results, feature_names)
            print(f"\n✅ Modèle sauvegardé avec succès!")
            print(f"📁 Fichier modèle: {model_file}")
            print(f"📄 Fichier métadonnées: {metadata_file}")
        
        return results, best_model

if __name__ == "__main__":
    predictor = CovidMeteoPredictor()
    results, best_model = predictor.run_analysis(
        'data/owid-covid-data_cleaned.csv',
        'data/fusion_meteo_3.csv',
        save_model=True
    ) 