import ray
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ray.init(ignore_reinit_error=True)

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

@ray.remote
def train_model(X_train, y_train, X_test, y_test, start_time, start_memory):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    current_time, current_memory, throughput = monitor_resources(start_time, start_memory)

    return accuracy, current_time, current_memory, throughput

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = generate_data()

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    future = train_model.remote(X_train, y_train, X_test, y_test, start_time, start_memory)
    results = ray.get(future)

    accuracy, elapsed_time, memory_used, throughput = results

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Memory used: {memory_used:.2f} GB")
    print(f"Throughput: {throughput:.4f} units/sec")

    metrics = {
        "Training Time (s)": elapsed_time,
        "Memory Used (GB)": memory_used,
        "Throughput (units/sec)": throughput
    }

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title("Performance Metrics")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd

spark = SparkSession.builder \
    .appName("SparkMLMonitoring") \
    .getOrCreate()

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3) 

    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

def create_spark_dataframe():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(1, 21)])
    df['label'] = y

    spark_df = spark.createDataFrame(df)
    return spark_df

def train_spark_model(spark_df, start_time, start_memory):
    feature_columns = [f'feature{i}' for i in range(1, 21)]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[assembler, lr])

    model = pipeline.fit(spark_df)

    current_time, current_memory, throughput = monitor_resources(start_time, start_memory)

    return model, current_time, current_memory, throughput
def main():
    spark_df = create_spark_dataframe()

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    model, elapsed_time, memory_used, throughput = train_spark_model(spark_df, start_time, start_memory)

    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Memory used: {memory_used:.2f} GB")
    print(f"Throughput: {throughput:.4f} units/sec")

    metrics = {
        "Training Time (s)": elapsed_time,
        "Memory Used (GB)": memory_used,
        "Throughput (units/sec)": throughput
    }

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title("Performance Metrics")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train_model(train_loader, model, criterion, optimizer, start_time, start_memory):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        current_time, current_memory, throughput = monitor_resources(start_time, start_memory)

    return current_time, current_memory, throughput

def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = generate_data()

    input_dim = 20  # Number of features in the dataset
    output_dim = 2  # Number of classes in the dataset
    model = SimpleNN(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    elapsed_time, memory_used, throughput = train_model(train_loader, model, criterion, optimizer, start_time, start_memory)

    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Memory used: {memory_used:.2f} GB")
    print(f"Throughput: {throughput:.4f} units/sec")

    metrics = {
        "Training Time (s)": elapsed_time,
        "Memory Used (GB)": memory_used,
        "Throughput (units/sec)": throughput
    }

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title("Performance Metrics")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import tensorflow as tf
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

def create_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, start_time, start_memory):
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)  # Set verbose to 0 to suppress output
    current_time, current_memory, throughput = monitor_resources(start_time, start_memory)
    return current_time, current_memory, throughput

def main():
    X_train, X_test, y_train, y_test = generate_data()

    input_dim = X_train.shape[1]  # Number of features
    output_dim = len(np.unique(y_train))  # Number of classes
    model = create_model(input_dim, output_dim)

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    elapsed_time, memory_used, throughput = train_model(model, X_train, y_train, start_time, start_memory)

    print(f"Training time: {elapsed_time:.2f} seconds")
    print(f"Memory used: {memory_used:.2f} GB")
    print(f"Throughput: {throughput:.4f} units/sec")

    metrics = {
        "Training Time (s)": elapsed_time,
        "Memory Used (GB)": memory_used,
        "Throughput (units/sec)": throughput
    }

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title("Performance Metrics")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB
    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train_model(train_loader, model, criterion, optimizer, start_time, start_memory):
    model.train()
    times, memory, throughputs = [], [], []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        current_time, current_memory, throughput = monitor_resources(start_time, start_memory)

        times.append(current_time)
        memory.append(current_memory)
        throughputs.append(throughput)
    return times, memory, throughputs

def main():
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = generate_data()

    input_dim = 20  # Number of features in the dataset
    output_dim = 2  # Number of classes in the dataset
    model = SimpleNN(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    times, memory_used, throughput = train_model(train_loader, model, criterion, optimizer, start_time, start_memory)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(times, label="Training Time (s)")
    plt.title("Training Time")
    plt.xlabel("Iterations")
    plt.ylabel("Time (s)")

    plt.subplot(1, 3, 2)
    plt.plot(memory_used, label="Memory Used (GB)", color='orange')
    plt.title("Memory Usage")
    plt.xlabel("Iterations")
    plt.ylabel("Memory (GB)")

    plt.subplot(1, 3, 3)
    plt.plot(throughput, label="Throughput (units/sec)", color='green')
    plt.title("Throughput")
    plt.xlabel("Iterations")
    plt.ylabel("Throughput (units/sec)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import tensorflow as tf
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB
    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

def create_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, start_time, start_memory):
    times, memory, throughputs = [], [], []
    for epoch in range(5):
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

        current_time, current_memory, throughput = monitor_resources(start_time, start_memory)

        times.append(current_time)
        memory.append(current_memory)
        throughputs.append(throughput)

    return times, memory, throughputs

def main():
    X_train, _, y_train, _ = generate_data()

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    model = create_model(input_dim, output_dim)

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)

    times, memory_used, throughput = train_model(model, X_train, y_train, start_time, start_memory)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(times, label="Training Time (s)")
    plt.title("Training Time")
    plt.xlabel("Epochs")
    plt.ylabel("Time (s)")

    plt.subplot(1, 3, 2)
    plt.plot(memory_used, label="Memory Used (GB)", color='orange')
    plt.title("Memory Usage")
    plt.xlabel("Epochs")
    plt.ylabel("Memory (GB)")

    plt.subplot(1, 3, 3)
    plt.plot(throughput, label="Throughput (units/sec)", color='green')
    plt.title("Throughput")
    plt.xlabel("Epochs")
    plt.ylabel("Throughput (units/sec)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import ray
import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ray import remote

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB
    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

def create_model(input_dim, output_dim):
    # Simple dummy function to simulate model training
    def model(data):
        time.sleep(0.1)  # Simulate some computation time
        return np.random.rand(output_dim)
    return model

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@remote
def train_model_ray(model, data, labels):
    time.sleep(0.1)  
    return model(data), labels

def main():
    ray.init(ignore_reinit_error=True)  # Initialize Ray

    X_train, _, y_train, _ = generate_data()

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))

    model = create_model(input_dim, output_dim)

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    times, memory_used, throughput = [], [], []
    num_epochs = 5

    for epoch in range(num_epochs):
        future = train_model_ray.remote(model, X_train, y_train)
        results = ray.get(future)

        current_time, current_memory, throughput_val = monitor_resources(start_time, start_memory)

        times.append(current_time)
        memory_used.append(current_memory)
        throughput.append(throughput_val)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(times, label="Training Time (s)")
    plt.title("Training Time")
    plt.xlabel("Epochs")
    plt.ylabel("Time (s)")

    plt.subplot(1, 3, 2)
    plt.plot(memory_used, label="Memory Used (GB)", color='orange')
    plt.title("Memory Usage")
    plt.xlabel("Epochs")
    plt.ylabel("Memory (GB)")

    plt.subplot(1, 3, 3)
    plt.plot(throughput, label="Throughput (units/sec)", color='green')
    plt.title("Throughput")
    plt.xlabel("Epochs")
    plt.ylabel("Throughput (units/sec)")

    plt.tight_layout()
    plt.show()

    ray.shutdown()  # Shutdown Ray

if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import pandas as pd

def monitor_resources(start_time, start_memory):
    current_time = time.time() - start_time
    current_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB
    throughput = current_time / (current_memory - start_memory) if current_memory - start_memory != 0 else 0
    return current_time, current_memory, throughput

spark = SparkSession.builder.appName("SparkResourceMonitoring").getOrCreate()

def generate_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(20)])
    df['label'] = y
    return spark.createDataFrame(df)

def train_model(df, start_time, start_memory):
    assembler = VectorAssembler(inputCols=[f'feature{i}' for i in range(20)], outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[assembler, lr])

    model = pipeline.fit(df)

    current_time, current_memory, throughput = monitor_resources(start_time, start_memory)

    return current_time, current_memory, throughput

def main():
    df = generate_data()

    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 ** 3)  # GB

    times, memory_used, throughput = [], [], []

    num_epochs = 10  # Simulate multiple training epochs

    for epoch in range(num_epochs):
        elapsed_time, memory_val, throughput_val = train_model(df, start_time, start_memory)

        times.append(elapsed_time)
        memory_used.append(memory_val)
        throughput.append(throughput_val)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(times, label="Training Time (s)", color='blue')
    plt.title("Training Time")
    plt.xlabel("Epochs")
    plt.ylabel("Time (s)")

    plt.subplot(1, 3, 2)
    plt.plot(memory_used, label="Memory Used (GB)", color='orange')
    plt.title("Memory Usage")
    plt.xlabel("Epochs")
    plt.ylabel("Memory (GB)")

    plt.subplot(1, 3, 3)
    plt.plot(throughput, label="Throughput (units/sec)", color='green')
    plt.title("Throughput")
    plt.xlabel("Epochs")
    plt.ylabel("Throughput (units/sec)")

    plt.tight_layout()
    plt.show()

    spark.stop()  # Shutdown Spark session

if __name__ == "__main__":
    main()