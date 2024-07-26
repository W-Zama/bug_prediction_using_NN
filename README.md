# bug_prediction_using_NN
Neural Netwotkを使って，フォールト数を予測します．

## メモ

### 単調増加性の確保

すべての重みを0以上にする制約を加えることで実現．

```python
from tensorflow.keras.constraints import NonNeg
model.add(Dense(128, activation='tanh', kernel_constraint=NonNeg()))
```

