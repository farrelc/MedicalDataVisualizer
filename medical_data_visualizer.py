import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = 1 if df['weight'].all()/(df['height'].all()/100)**(2) > 25 else  0

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df[['cholesterol', 'gluc']] = np.where(df[['cholesterol', 'gluc']].values > 1, 1, 0)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc','smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = pd.melt(df, value_vars=["active", "alco", "cholesterol", "gluc", "overweight", "smoke"], id_vars="cardio")
    df_cat['value'] = df_cat['value'].astype(int)

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x = 'variable', hue = 'value', col = 'cardio', data = df_cat, kind = 'count')
    fig.set_ylabels('total')
    fig = fig.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    diastolic = df['ap_lo'] <= df['ap_hi']
    hl = df['height'] >= df['height'].quantile(0.025)
    hh = df['height'] <= df['height'].quantile(0.975)
    wl = df['weight'] >= df['weight'].quantile(0.025)
    wh = df['weight'] <= df['weight'].quantile(0.975)
    df_heat = df[diastolic & hl & hh & wl & wh]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(13, 10))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, linewidths=.5, vmin = -.16, vmax = .32, center=0, fmt=".1f", cbar_kws = {'shrink':0.5})

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
