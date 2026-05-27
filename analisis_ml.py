"""
Clasificación binaria normal vs encefalopatía mediante aprendizaje automático.

Comparación de Random Forest, LR-L1 y Linear SVM mediante
Leave-One-Out CV con grid search anidado (3-fold estratificado interno).

Entrada: features_ml.csv con 131 columnas de features.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') #evita 'Tcl_AsyncDelete' y 'main thread is not in main loop'.
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, LeaveOneOut, StratifiedKFold, permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import shap

RANDOM_STATE = 42
N_PERMUTATIONS = 1000
N_JOBS = 3  # núcleos a usar en paralelo
# Umbral de correlación para agrupar features en clusters jerárquicos en el análisis SHAP del Random Forest
UMBRAL_CORR_SHAP = 0.7 # (0.7 = fuertemente correlacionadas).

DIR_RESULTADOS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Resultados')
CSV_FEATURES = os.path.join(DIR_RESULTADOS, 'features_ml.csv')


# Carga features_ml.csv: normal -> 0, encefalopatia{1,2} -> 1.
def cargar_dataset(ruta_csv):
    datos = pd.read_csv(ruta_csv)
    datos = datos[datos['etiqueta'] != 'desconocido'].copy()
    # .startswith('encefalopatia') agrupa encefalopatia1 y encefalopatia2.
    datos['y'] = datos['etiqueta'].str.startswith('encefalopatia').astype(int)

    cols_features = [c for c in datos.columns if c not in ('archivo', 'etiqueta', 'y')]
    X = datos[cols_features].to_numpy(dtype=float) # todas las columnas de features_ml.csv excepto archivo, etiqueta e y
    y = datos['y'].to_numpy(dtype=int) # columna binarizada: 0 normal, 1 encefalopatía
    archivos = datos['archivo'].tolist()
    return X, y, cols_features, archivos


# Devuelve un diccionario con los 3 clasificadores que se van a comparar, cada uno con su pipeline y su grid
# de hiperparámetros para el grid search.
def definir_modelos():
    return {
        'random_forest': {
            'pipe': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=1))
            ]),
            'grid': {
                'clf__n_estimators': [200, 500],
                'clf__max_depth': [None, 3, 5],
                'clf__min_samples_leaf': [1, 2, 4],
            },
        },
        'lr_l1': {
            'pipe': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(l1_ratio=1, solver='liblinear', class_weight='balanced', # liblinear más rapido y eficiente
                    random_state=RANDOM_STATE, max_iter=5000))
            ]),
            'grid': {'clf__C': [0.01, 0.1, 1.0, 10.0]},
        },
        'svm_lineal': {
            'pipe': Pipeline([
                ('scaler', StandardScaler()),
                # El SVM lineal se calibra mediante Platt scaling (CalibratedClassifierCV, cv=3) para obtener probabilidades
                # comparables a RF y LR.
                ('clf', CalibratedClassifierCV(estimator=LinearSVC(class_weight='balanced',
                        random_state=RANDOM_STATE,
                        max_iter=10000,
                        dual='auto',
                    ),
                    cv=3,
                    method='sigmoid',
                )),
            ]),
            'grid': {'clf__estimator__C': [0.01, 0.1, 1.0, 10.0]},
        },
    }


# Crea (si no existe) y devuelve la ruta de la carpeta donde se guardarán todos los resultados de un modelo
# concreto.
def _carpeta_modelo(nombre):
    ruta = os.path.join(DIR_RESULTADOS, f'ml_{nombre}')
    os.makedirs(ruta, exist_ok=True)
    return ruta


# Se usa para agrupar features correlacionadas en el análisis SHAP del Random Forest.
def _clusters_correlacion(X, umbral_corr):
    corr = np.corrcoef(X.T)
    # Distancia 1-|r|: features muy correlacionadas (|r|->1) quedan a distancia 0.
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    np.clip(dist, 0, None, out=dist) # Eliminamos negativos.
    enlace_jerarquico = linkage(squareform(dist, checks=False), method='average')
    # Pares con distancia <= t (correlacionadas) se funden.
    return fcluster(enlace_jerarquico, t=1 - umbral_corr, criterion='distance')


# Hace la evaluación de un modelo usando validación cruzada anidada: por fuera Leave-One-Out (LOO)
# y por dentro un grid search con 3-fold estratificado. Devuelve las 22 predicciones agregadas para
# calcular métricas globales.
def _evaluar_loo_anidado(pipeline, grid, X, y):
    cv_externo = LeaveOneOut()
    cv_interno = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    y_pred = np.zeros_like(y)
    y_proba = np.zeros(len(y), dtype=float)

    for idx_train, idx_test in cv_externo.split(X):
        grid_search = GridSearchCV(pipeline, grid, cv=cv_interno, scoring='roc_auc', n_jobs=N_JOBS, refit=True)
        grid_search.fit(X[idx_train], y[idx_train])
        y_pred[idx_test] = grid_search.predict(X[idx_test])
        y_proba[idx_test] = grid_search.predict_proba(X[idx_test])[:, 1]

    return y_pred, y_proba


# Calcula las 4 métricas de rendimiento del modelo a partir de las predicciones LOO y las devuelve
# en un diccionario.
def _metricas_globales(y, y_pred, y_proba):
    return {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_acc': balanced_accuracy_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_proba),
    }


# Reentrena el pipeline usando los 22 sujetos completos y elige los mejores hiperparámetros mediante grid search
# 3-fold estratificado
def _entrenar_modelo_final(pipeline, grid, X, y):
    # Atajo para el dummy_stratified: sin grid no tiene sentido lanzar GridSearchCV.
    if not grid:
        pipeline.fit(X, y)
        return pipeline, {}

    grid_search = GridSearchCV(pipeline, grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring='roc_auc', n_jobs=N_JOBS, refit=True)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_


# Extrae un vector de importancia por feature comparable entre los 3 modelos.
# RF: importancia Gini. LR L1: |coef|. SVM lineal calibrado: |coef| promedio
# entre los estimadores internos de la calibración.
def _importancias_modelo(nombre, pipeline_final):
    clasificador = pipeline_final.named_steps['clf']
    if nombre == 'random_forest':
        return clasificador.feature_importances_, 'importancia Gini'
    if nombre == 'lr_l1':
        return np.abs(clasificador.coef_[0]), '|coef|'
    if nombre == 'svm_lineal':
        coefs = np.vstack([
            cc.estimator.coef_[0] for cc in clasificador.calibrated_classifiers_
        ])
        return np.mean(np.abs(coefs), axis=0), '|coef| medio (calibrados)'
    return None, None


# Permutation test con 5-fold estratificado y N_PERMUTATIONS.
# Usa los hiperparámetros ya escogidos en el refit final (pipeline_final).
def _permutation_test_modelo(pipeline_final, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_real, scores_nulos, p_value = permutation_test_score(pipeline_final, X, y, scoring='roc_auc',
        cv=cv, n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    # auc_perm_ref: AUC del pipeline_final evaluado con 5-fold sobre todo el dataset. Es el AUC que
    # se contrasta contra la distribución nula del permutation test.
    return {
        'auc_perm_ref': float(auc_real),
        'auc_null_mean': float(np.mean(scores_nulos)),
        'auc_null_std': float(np.std(scores_nulos)),
        'p_value': float(p_value),
    }, scores_nulos


# Genera y guarda la confusion matrix del modelo a partir de las predicciones LOO.
def _guardar_confusion(y, y_pred, nombre, ruta):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    matriz_confusion = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(matriz_confusion, display_labels=['normal', 'encefalopatía']).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion matrix LOO — {nombre}')
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# Genera y guarda la curva ROC del modelo a partir de las probabilidades LOO.
def _guardar_roc(y, y_proba, auc, nombre, ruta):
    fpr, tpr, _ = roc_curve(y, y_proba)  # fpr: tasa de falsos positivos, tpr: tasa de verdaderos positivos
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set_xlabel('Tasa de falsos positivos')
    ax.set_ylabel('Tasa de verdaderos positivos')
    ax.set_title(f'Curva ROC LOO — {nombre}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# Genera y guarda un bar plot horizontal con las 15 features más importantes del modelo.
def _guardar_top_features(importancias, etiqueta, nombres_features, nombre, ruta, k=15):
    idx_top = np.argsort(importancias)[::-1][:k]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(range(len(idx_top)), importancias[idx_top][::-1], color='steelblue')
    ax.set_yticks(range(len(idx_top)))
    ax.set_yticklabels([nombres_features[i] for i in idx_top[::-1]], fontsize=8)
    ax.set_xlabel(etiqueta)
    ax.set_title(f'Top-{k} features — {nombre}')
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# Genera y guarda un histograma del test de permutación, que sirve para evaluar si el AUC real del modelo es
# estadísticamente significativo o se podría haber obtenido por azar.
def _guardar_histograma_permutacion(auc_real, scores_nulos, p_value, nombre, ruta):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(scores_nulos, bins=30, color='lightgray', edgecolor='gray', label=f'AUC nulo (n={len(scores_nulos)})')
    ax.axvline(auc_real, color='crimson', linewidth=2, label=f'AUC real = {auc_real:.3f}')
    ax.set_xlabel('AUC')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Permutation test — {nombre}  (p = {p_value:.4f})')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# Calcula y dibuja la permutation importance de cada feature para el Random Forest, usando un esquema de
# validación cruzada honesta (sobre datos no vistos). La uso para contrastar con el análisis SHAP, que
# es más robusto a la correlación entre features.
def _permutation_importance_rf(pipeline_final, X, y, nombres_features, dir_modelo, n_repeats=30, top_k=15):
    print(' a) Calculando permutation importance por fold...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    importancias_por_fold = []
    for idx_train, idx_test in cv.split(X, y):
        pipeline_fold = clone(pipeline_final) # Mismos hiperparámetros, pero sin entrenar.
        pipeline_fold.fit(X[idx_train], y[idx_train])
        resultado = permutation_importance(pipeline_fold, X[idx_test], y[idx_test], scoring='roc_auc',
            n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=N_JOBS)
        importancias_por_fold.append(resultado.importances_mean)

    importancias = np.vstack(importancias_por_fold)
    means = importancias.mean(axis=0)
    stds = importancias.std(axis=0)
    idx_top = np.argsort(means)[::-1][:top_k] # Devuelve un array de enteros de índices.

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y_pos = range(len(idx_top))
    ax.barh(y_pos, means[idx_top][::-1], xerr=stds[idx_top][::-1], color='darkorange',
        ecolor='gray', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([nombres_features[i] for i in idx_top[::-1]], fontsize=8)
    ax.set_xlabel(f'Caída media de AUC al barajar (5-fold, n_repeats={n_repeats})')
    ax.set_title(f'Permutation importance — Random Forest (top-{top_k})')
    ax.axvline(0, color='black', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_modelo, 'permutation_importance.png'), dpi=200)
    plt.close(fig)


# TreeSHAP + clustering de features correlacionadas (sólo Random Forest).
def _analizar_shap_rf(pipeline_final, X, y, archivos, nombres_features, dir_modelo,
                     y_proba_loo, umbral_corr=UMBRAL_CORR_SHAP, top_k_features=15, top_k_clusters=7):

    print(' b) Calculando SHAP (TreeSHAP)...')

    scaler = pipeline_final.named_steps['scaler']
    rf = pipeline_final.named_steps['clf']
    X_scaled = scaler.transform(X) # Estandariza X (media 0, std 1).

    explainer = shap.TreeExplainer(rf)
    # shap_values devuelve (n_pacientes, n_features, n_clases); [:, :, 1] = SHAP de la clase positiva (encefalopatía).
    shap_pos = explainer.shap_values(X_scaled)[:, :, 1]
    importancia_global = np.abs(shap_pos).mean(axis=0)

    # Gráfico Beeswarm con las top_15 features más importantes por SHAP.
    idx_top = np.argsort(importancia_global)[::-1][:top_k_features] # Índices de las top_k features ordenadas por importancia descendente.
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_pos[:, idx_top], X_scaled[:, idx_top], feature_names=[nombres_features[i] for i in idx_top], show=False)
    plt.title(f'SHAP beeswarm — top-{top_k_features} features (RF)')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_modelo, 'shap_beeswarm.png'), dpi=200, bbox_inches='tight')
    plt.close('all')

    clusters = _clusters_correlacion(X, umbral_corr)
    cluster_imp = {}
    cluster_features = {}
    for cluster_id in np.unique(clusters):
        idx = np.where(clusters == cluster_id)[0] # Índices (posiciones en X) de las features que pertenecen a este cluster.
        cluster_imp[cluster_id] = float(importancia_global[idx].sum()) # Importancia cluster = suma de |SHAP| medio de sus features.
        cluster_features[cluster_id] = [nombres_features[i] for i in idx] # Nombres de las features del cluster.

    clusters_orden = sorted(cluster_imp.items(), key=lambda kv: kv[1], reverse=True) # Ordena los clusters por importancia descendente.
    top_clusters = clusters_orden[:top_k_clusters]

    etiquetas = []
    valores = []
    # Etiquetamos cada barra con la feature individual de mayor |SHAP| dentro del cluster + cuantas le acompañan (+N).
    for cluster_id, imp in top_clusters:
        features_cluster = cluster_features[cluster_id]
        feature_dominante = features_cluster[int(np.argmax([importancia_global[nombres_features.index(f)] for f in features_cluster]))]
        etiquetas.append(f'{feature_dominante} (+{len(features_cluster) - 1})' if len(features_cluster) > 1 else feature_dominante)
        valores.append(imp)

    # Gráfico de barras horizontal con los top_7 clusters de features correlacionadas.
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = range(len(etiquetas))
    ax.barh(y_pos, valores[::-1], color='teal') # [::-1] invierte el orden para que la barra más importante quede arriba.
    ax.set_yticks(y_pos)
    ax.set_yticklabels(etiquetas[::-1], fontsize=8)
    ax.set_xlabel('|SHAP| medio sumado en el grupo')
    ax.set_title(f'SHAP por grupo de features correlacionadas (|r|>={umbral_corr})')
    fig.tight_layout()
    fig.savefig(os.path.join(dir_modelo, 'shap_grupos.png'), dpi=200)
    plt.close(fig)

    # Dependence plot para la feature individual más importante. Se compara valor de la feature y su SHAP.
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(int(idx_top[0]), shap_pos, X_scaled, feature_names=nombres_features, interaction_index=None, show=False)
    plt.title(f'SHAP dependence — {nombres_features[idx_top[0]]}')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_modelo, 'shap_dependence_top1.png'),
                dpi=200, bbox_inches='tight')
    plt.close('all')

    # Estudio los 2 normales en LOO (más confundidos como enc) y las 2 encefalopatías en LOO (más confundidas como normal).
    idx_normales = np.where(y == 0)[0]
    idx_encefalopatia = np.where(y == 1)[0]
    idx_normales_peores = idx_normales[np.argsort(y_proba_loo[idx_normales])[::-1][:2]]
    idx_enc_peores = idx_encefalopatia[np.argsort(y_proba_loo[idx_encefalopatia])[:2]]

    # expected_value[1] = clase positiva (encefalopatía).
    valor_base_shap = explainer.expected_value[1]

    # Lista de pacientes para hacer decision plots: (etiqueta_clase, idx, p(enc) LOO, descripción).
    pacientes = []
    for rank, idx in enumerate(idx_normales_peores, 1):
        idx = int(idx)
        pacientes.append(('normal_peor', idx, y_proba_loo[idx], f'#{rank} peor LOO'))
    for rank, idx in enumerate(idx_enc_peores, 1):
        idx = int(idx)
        pacientes.append(('encefalopatia_peor', idx, y_proba_loo[idx], f'#{rank} peor LOO'))

    for clase, idx, proba_ref, etiqueta in pacientes:
        plt.figure(figsize=(7, 6))
        # decision_plot: trayectoria del base_value hasta la predicción del paciente, sumando SHAP feature a feature.
        shap.decision_plot(valor_base_shap, shap_pos[idx, :], X_scaled[idx, :], feature_names=nombres_features,
            feature_display_range=slice(None, -top_k_features - 1, -1), show=False) # Se muestran solo las top_k_features.
        plt.title(f'SHAP decision — {archivos[idx]} ({clase}, {etiqueta})\np(enc) = {proba_ref:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_modelo, f'shap_paciente_{clase}_{archivos[idx]}.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

    # Resumen con todos los clusters.
    resumen = pd.DataFrame([
        {
            'cluster_id': int(cluster_id),
            'n_features': len(cluster_features[cluster_id]),
            'importancia_shap_sum': cluster_imp[cluster_id],
            'features': '; '.join(cluster_features[cluster_id]),
        }
        for cluster_id, _ in clusters_orden
    ])
    resumen.to_csv(os.path.join(dir_modelo, 'shap_clusters.csv'), index=False)

    print(f'   SHAP: {len(np.unique(clusters))} clusters al umbral |r|>={umbral_corr}')
    print('   Decision plots generados para los 4 sujetos peor clasificados en LOO:')
    for clase, idx, proba_ref, _ in pacientes:
        print(f'    - {clase:20s} {archivos[idx]:30s} p(enc)_LOO={proba_ref:.3f}')


# Evalúa un modelo con LOO, lo reentrena con todo el dataset. Devuelve métricas + pipeline final.
def procesar_modelo(nombre, config, X, y, nombres_features):
    print(f'\n- Evaluando {nombre} (LOO externo + grid 3-fold interno)...')
    y_pred, y_proba = _evaluar_loo_anidado(config['pipe'], config['grid'], X, y)
    metricas = _metricas_globales(y, y_pred, y_proba)
    print(f'  acc={metricas["accuracy"]:.3f}  bal_acc={metricas["balanced_acc"]:.3f}  f1={metricas["f1"]:.3f}  '
        f'auc={metricas["auc"]:.3f}')

    dir_modelo = _carpeta_modelo(nombre)
    # Guarda la confusion matrix LOO y curva ROC como PNG.
    _guardar_confusion(y, y_pred, nombre, os.path.join(dir_modelo, 'confusion.png'))
    _guardar_roc(y, y_proba, metricas['auc'], nombre, os.path.join(dir_modelo, 'roc.png'))

    pipeline_final, mejores_hp = _entrenar_modelo_final(config['pipe'], config['grid'], X, y)

    # Extrae el vector de importancia por feature (Gini para RF, |coef| para LR/SVM lineal). Dummy no tiene importancias.
    importancias, etiqueta_imp = _importancias_modelo(nombre, pipeline_final)
    if importancias is not None:
        _guardar_top_features(importancias, etiqueta_imp, nombres_features, nombre, os.path.join(dir_modelo, 'top_features.png'))

    # El dummy es azar por construcción: ahorramos las 1000 permutaciones x 5-fold.
    if nombre != 'dummy_stratified':
        print(f'  Permutation test 5-fold x {N_PERMUTATIONS} permutaciones:')
        res_perm, distribucion_nula = _permutation_test_modelo(pipeline_final, X, y)
        metricas.update(res_perm) # Añade AUC real, AUC nulo medio/std y p-value al diccionario de métricas.

        # Guarda el histograma de la distribución nula con la línea del AUC observado.
        _guardar_histograma_permutacion(res_perm['auc_perm_ref'], distribucion_nula, res_perm['p_value'], nombre,
            os.path.join(dir_modelo, 'permutation_test.png'))

        print(
            f'  auc_perm_ref={res_perm["auc_perm_ref"]:.3f}  '
            f'auc_null={res_perm["auc_null_mean"]:.3f}±{res_perm["auc_null_std"]:.3f}  '
            f'p={res_perm["p_value"]:.4f}'
        )

    # Imprime los hiperparámetros elegidos por el grid search.
    if mejores_hp:
        print('  Hiperparámetros:')
        for k, v in mejores_hp.items():
            print(f'    {k} = {v}')
    else:
        print('  Hiperparámetros: (sin grid)')

    # Guardo el pipeline entrenado en la carpeta del modelo. El dummy no aporta nada cargarlo.
    if nombre != 'dummy_stratified':
        dump(
            {
                'nombre': nombre,
                'pipeline': pipeline_final,
                'hiperparametros': mejores_hp,
                'features': nombres_features,
                'metricas_loo': metricas,
            },
            os.path.join(dir_modelo, 'modelo.joblib'),
        )

    return metricas, pipeline_final, mejores_hp, y_pred, y_proba


def main():
    t_inicio = time.perf_counter()

    if not os.path.exists(CSV_FEATURES):
        raise FileNotFoundError(
            f'No existe {CSV_FEATURES}. Ejecuta antes estudio_coherencia.py.'
        )

    X, y, nombres_features, archivos = cargar_dataset(CSV_FEATURES)

    n_encefalopatia = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    print(f'Registros: N={len(y)}  (normal={n_normal}, encefalopatía={n_encefalopatia})')
    print(f'Features: {X.shape[1]}')

    # Baseline aleatorio estratificado. Para comparar contra modelos reales y verificar que superan esta referencia inferior.
    dummy_config = {
        'pipe': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)),
        ]),
        'grid': {},
    }
    # Evalúa el dummy con LOO y guarda sus métricas como referencia inferior.
    metricas_dummy, _, _, _, _ = procesar_modelo(
        'dummy_stratified', dummy_config, X, y, nombres_features
    )

    # Diccionarios para acumular resultados de los 3 modelos reales + dummy.
    resultados = {'dummy_stratified': metricas_dummy}
    pipelines_finales = {}
    proba_loo = {}

    # Itera sobre los 3 clasificadores reales (RF, LR L1, SVM lineal).
    modelos = definir_modelos()
    for nombre, config in modelos.items():
        metricas, pipeline_final, _, _, y_proba = procesar_modelo(nombre, config, X, y, nombres_features)
        resultados[nombre] = metricas
        pipelines_finales[nombre] = pipeline_final
        proba_loo[nombre] = y_proba

    tabla_resumen = pd.DataFrame(resultados).T.round(3)

    # Para hacer más legible la tabla.
    t = tabla_resumen.copy()
    t['auc_null'] = [f'{m:.3f} ± {s:.3f}' if pd.notna(m) else '—'
                     for m, s in zip(t['auc_null_mean'], t['auc_null_std'])]
    t = t.drop(columns=['auc_null_mean', 'auc_null_std']).fillna('—')
    t = t[['accuracy', 'balanced_acc', 'f1', 'auc', 'auc_perm_ref', 'auc_null', 'p_value']]

    print('\n                                                     === Resumen Modelos ===')
    print(t.to_string(col_space=16, justify='center'))

    # Shap y permutation importance sólo para el Random Forest.
    if 'random_forest' in pipelines_finales:
        dir_rf = _carpeta_modelo('random_forest')
        print('\nPermutation importance + SHAP — Random Forest:')
        _permutation_importance_rf(pipelines_finales['random_forest'], X, y, nombres_features, dir_rf)
        _analizar_shap_rf(pipelines_finales['random_forest'], X, y, archivos, nombres_features, dir_rf,
            y_proba_loo=proba_loo.get('random_forest'))

    t_total = time.perf_counter() - t_inicio
    horas, resto = divmod(t_total, 3600)
    minutos, segundos = divmod(resto, 60)
    print(f'\nTiempo total de ejecución: {t_total:.2f} s ({int(horas):02d}:{int(minutos):02d}:{segundos:05.2f})')


if __name__ == '__main__':
    main()