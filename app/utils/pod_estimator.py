import pandas as pd

def predict_pods(tps, cpu, mem, response, cpu_model, mem_model, cpu_limit, mem_limit):
    for pods in range(1, 50):
        # Use full TPS for prediction (do NOT divide by pods)
        sample = pd.DataFrame([[tps, cpu, mem, response]],
                              columns=["TPS", "CPU_Cores", "Memory_GB", "ResponseTime_sec"])

        cpu_pred = cpu_model.predict(sample)[0]
        mem_pred = mem_model.predict(sample)[0]

        # Calculate per pod usage
        cpu_per_pod = cpu_pred / pods
        mem_per_pod = mem_pred / pods

        if cpu_per_pod <= cpu_limit and mem_per_pod <= mem_limit:
            return pods, cpu_per_pod, mem_per_pod, "✅ Configuration is within acceptable limits."

    return 49, cpu_pred / 49, mem_pred / 49, "⚠️ Configuration exceeds limits. Not recommended for production."
