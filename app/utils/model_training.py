from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_models(df):
    features = ["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_sec"]
    cpu_target = "CPU_Load"
    mem_target = "Memory_Load"

    X = df[features]
    y_cpu = df[cpu_target]
    y_mem = df[mem_target]

    X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(X, y_cpu, test_size=0.2, random_state=42)
    _, _, y_mem_train, y_mem_test = train_test_split(X, y_mem, test_size=0.2, random_state=42)

    cpu_model = LinearRegression().fit(X_train, y_cpu_train)
    mem_model = LinearRegression().fit(X_train, y_mem_train)

    cpu_r2 = r2_score(y_cpu_test, cpu_model.predict(X_test))
    mem_r2 = r2_score(y_mem_test, mem_model.predict(X_test))

    return cpu_model, mem_model, cpu_r2, mem_r2
