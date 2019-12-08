import pathlib
import os
import plotnine as gg
import pandas as pd

path_bank_df = pathlib.Path("../data/raw/bank-full.csv")

bank_df = pd.read_csv(path_bank_df, delimiter=";")

plot_balance = gg.ggplot(bank_df, gg.aes(x="balance"))+ gg.geom_histogram()
plot_duration = gg.ggplot(bank_df, gg.aes(x="duration"))+ gg.geom_histogram()

plot_balance.save
plot_duration.save

plot_ba_bw = plot_balance + gg.theme_bw()
plot_du_bw = plot_duration + gg.theme_bw()

plot_ba_bw.save(pathlib.Path("../reports/figures/hist_balance_bw.png"))
plot_du_bw.save(pathlib.Path("../reports/figures/hist_duration_bw.png"))