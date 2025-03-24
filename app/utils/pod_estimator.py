def predict_pods(tps, cpu, mem, resp, cpu_model, mem_model, cpu_thresh, mem_thresh):
    for pods in range(1, 50):
        avg_tps = tps / pods
        sample = [[avg_tps, cpu, mem, resp]]

        pred_cpu = cpu_model.predict(sample)[0]
        pred_mem = mem_model.predict(sample)[0]

        if pred_cpu <= cpu_thresh and pred_mem <= mem_thresh:
            return pods, pred_cpu, pred_mem, "✅ Configuration is within acceptable limits."

    return 49, pred_cpu, pred_mem, "⚠️ Configuration exceeds limits. Not recommended for production."
