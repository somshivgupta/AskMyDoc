def precision_at_k(results, ground_truth_keywords, k=3):
    relevant = 0
    
    for r in results[:k]:
        if any(keyword in r.lower() for keyword in ground_truth_keywords):
            relevant += 1
    
    return relevant / k