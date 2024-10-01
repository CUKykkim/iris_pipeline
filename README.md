# 분석 파이프라인으로 iris 분류하기



## 노트북 생성

- 아래 이미지의 노트북을 생성한다. 

```
kubeflownotebookswg/jupyter-scipy:v1.9.0
```


## 전처리

```
from kfp.v2.dsl import component, Input, Output, Dataset

@component(
    base_image='kubeflownotebookswg/jupyter-tensorflow-full:v1.9.0'  # 기본 이미지 설정
)
def preprocess_iris_data(X_train_output: Output[Dataset], X_test_output: Output[Dataset], 
                         y_train_output: Output[Dataset], y_test_output: Output[Dataset]):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    # 데이터 로드 및 전처리
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 데이터를 파일로 저장 (각 출력 경로에)
    np.savetxt(X_train_output.path, X_train)
    np.savetxt(X_test_output.path, X_test)
    np.savetxt(y_train_output.path, y_train)
    np.savetxt(y_test_output.path, y_test)
```


### 학습

```
from kfp.v2.dsl import component, Input, Output, Dataset, Model

@component(
    base_image='kubeflownotebookswg/jupyter-tensorflow-full:v1.9.0'  # 기본 이미지 설정
)
def train_iris_model(X_train: Input[Dataset], y_train: Input[Dataset], model_output: Output[Model], epochs: int, batch_size: int):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    import numpy as np
    import tensorflow as tf
    
    # 데이터를 로드
    X_train_data = np.loadtxt(X_train.path)
    y_train_data = np.loadtxt(y_train.path)
    
    # 모델 정의
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_data.shape[1],)),
        Dense(3, activation='softmax')
    ])
    
    # 모델 컴파일 및 학습
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_data, y_train_data, epochs=epochs, batch_size=batch_size)
    
    # 훈련된 모델을 지정된 경로에 저장
    model.save(model_output.path)  # model_output.path는 모델이 저장될 경로
```


## 평가 

```
from kfp.v2.dsl import component, Input, Model, Output, Metrics


@component(
    base_image='kubeflownotebookswg/jupyter-tensorflow-full:v1.9.0'  # 기본 이미지 설정
)
def evaluate_iris_model(model_input: Input[Model], X_test_input: Input[Dataset], 
                        y_test_input: Input[Dataset], metrics: Output[Metrics]):

    import tensorflow as tf
    import numpy as np
    # 데이터를 파일에서 불러오기
    X_test = np.loadtxt(X_test_input.path)
    y_test = np.loadtxt(y_test_input.path)

    # 모델 로드
    model = tf.keras.models.load_model(model_input.path)

    # 테스트 데이터로 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    
    # 평가 결과 출력
    print(f"Test Accuracy: {accuracy}")
    
    # 메트릭 기록
    metrics.log_metric("accuracy", accuracy)
```

## 파이프라인 작성

```
from kfp.v2.dsl import pipeline
from kfp.v2 import compiler

@pipeline(name='iris-classification-pipeline')
def iris_pipeline(epochs: int = 10, batch_size: int = 32):
    # 1. 데이터 전처리 컴포넌트 실행
    preprocess_task = preprocess_iris_data()
    
    # 2. 학습 컴포넌트 실행
    train_task = train_iris_model(
        X_train=preprocess_task.outputs['X_train_output'],  # 전처리 컴포넌트의 출력이 학습 컴포넌트의 입력으로 사용
        y_train=preprocess_task.outputs['y_train_output'],  # 전처리 출력 연결
        epochs=epochs,
        batch_size=batch_size
    )
    
    # 3. 평가 컴포넌트 실행
    evaluate_task = evaluate_iris_model(
        model_input=train_task.outputs['model_output'],  # 학습 컴포넌트의 출력 (모델)을 평가 컴포넌트로 전달
        X_test_input=preprocess_task.outputs['X_test_output'],  # 전처리 컴포넌트의 테스트 데이터를 평가에 사용
        y_test_input=preprocess_task.outputs['y_test_output']
    )

# 파이프라인 컴파일
compiler.Compiler().compile(iris_pipeline, 'iris_pipeline.yaml')
```


## 하이퍼파라미터 최적화

```
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  namespace: kubeflow
  name: iris-hyperparameter-tuning
spec:
  objective:
    type: maximize 
    goal: 0.95 
    objectiveMetricName: accuracy  
  algorithm:
    algorithmName: random
  parallelTrialCount: 1
  maxTrialCount: 2
  maxFailedTrialCount: 2
  parameters:
    - name: epochs  # 하이퍼파라미터 epochs 정의
      parameterType: int
      feasibleSpace:
        min: "10"
        max: "100"
    - name: batchSize  # 하이퍼파라미터 batch_size 정의
      parameterType: int
      feasibleSpace:
        min: "16"
        max: "128"
  trialTemplate:
    retain: true
    primaryContainerName: training-container
    trialParameters:
      - name: epochs
        description: Number of epochs for training
        reference: epochs
      - name: batchSize
        description: Batch size for training
        reference: batchSize
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          metadata:
            annotations:
              sidecar.istio.io/inject: 'false'
          spec:
            containers:
              - name: training-container
                image: ykkim77/iris-train:latest  # 사용자 이미지로 변경
                command:
                  - "python"
                  - "/app/train.py"
                  - "--epochs=${trialParameters.epochs}"  # 파라미터 전달
                  - "--batch_size=${trialParameters.batchSize}"  # 파라미터 전달
                resources:
                  limits:
                    memory: "1Gi"
                    cpu: "1"
            restartPolicy: Never

```


## 모델 예측하기

- kserve로 엔드포인트 생성

```
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
```



- 내부에서 보안을 위해 api 호출을 제한했기 때문에 인증을 우회하는 정책을 적용한다. 

```
#allowlist-by-paths.yaml

apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allowlist-by-paths
  namespace: istio-system
spec:
  action: ALLOW
  rules:
  - to:
    - operation:
        paths:
        - /metrics
        - /healthz
        - /ready
        - /wait-for-drain
        - /v1/models/*
        - /v2/models/*
```

```
kubectl apply -f allowlist-by-paths.yaml
```


- 모델 예측 

```
import requests
import json

sklear_iris_input = dict(instances = [
        [6.8, 2.8, 4.8, 1.4],
        [6.0, 3.4, 4.5, 1.6],
        [5.0, 3.4, 1.5, 0.2], 
        [6.5, 3.0, 5.8, 2.2]
    ])

response_external = requests.post("http://sklearn-iris.kubeflow-user-example-com.svc.cluster.local/v1/models/sklearn-iris:predict", data = json.dumps(sklear_iris_input))   

print(response_external.text)
```