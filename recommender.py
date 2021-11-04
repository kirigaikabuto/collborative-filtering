from surprise import KNNWithMeans

sim_options = {
    "name": "cosine",
    "user_based": False,
}
algo = KNNWithMeans(sim_options=sim_options)
