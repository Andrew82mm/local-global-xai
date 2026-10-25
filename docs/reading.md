# Что читать

## Старт: книга Molnar, *Interpretable Machine Learning*

Бесплатно онлайн: https://christophm.github.io/interpretable-ml-book/

Главное по теме курсовой (названия глав могут отличаться в зависимости от издания):

1. Введение и глава про интерпретируемость: что такое локальные и глобальные объяснения.
2. Partial Dependence Plot (PDP) и Individual Conditional Expectation (ICE).
3. Accumulated Local Effects (ALE): обязательно разобраться, чем ALE лучше PDP
   при коррелированных признаках.
4. Permutation Feature Importance.
5. LIME (Local Surrogate).
6. Shapley Values, затем SHAP.

В каждой главе есть раздел Advantages и Disadvantages. Недостатки методов и есть
основной материал для главы «почему возникают расхождения».

## Оригинальные статьи (для списка литературы и главы 1)

- Ribeiro, Singh, Guestrin (2016). *"Why Should I Trust You?": Explaining the
  Predictions of Any Classifier*. Это LIME.
- Lundberg, Lee (2017). *A Unified Approach to Interpreting Model Predictions*. Это SHAP.
- Friedman (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Это PDP.
- Breiman (2001). *Random Forests*. Это Permutation Importance.
- Fisher, Rudin, Dominici (2019). *All Models are Wrong, but Many are Useful*.
  Model-agnostic Permutation Importance.
- Apley, Zhu (2020). *Visualizing the Effects of Predictor Variables in Black
  Box Supervised Learning Models*. Это ALE.

## Ближе всего к теме (читать после основ)

- Krishna et al. (2022). *The Disagreement Problem in Explainable Machine
  Learning: A Practitioner's Perspective*. Про расхождения объяснений и метрики
  их согласованности. Можно взять оттуда метрики.
- Hooker, Mentch, Zhou (2021). *Unrestricted Permutation Forces Extrapolation*.
  Почему PI и PDP врут при коррелированных признаках.
- Alvarez-Melis, Jaakkola (2018). *On the Robustness of Interpretability Methods*.
  Про нестабильность LIME.

## Документация библиотек

- SHAP: https://shap.readthedocs.io/
- LIME: https://github.com/marcotcr/lime
- scikit-learn, раздел Inspection (PDP, PI): https://scikit-learn.org/stable/inspection.html
- PyALE: https://github.com/DanaJomar/PyALE
