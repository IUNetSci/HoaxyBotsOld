from datetime import date
from datetime import timedelta
from os.path import join
import logging

from matplotlib import gridspec
from matplotlib import rcParams
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
import time
import json
import re
from os.path import dirname

from ccdf import ccdf
from ternary_heatmap_helper import heatmap_density, colorbar_hack
import os

logger = logging.getLogger(__name__)
# project root
BASE_DIR = dirname(dirname(os.path.realpath(__file__)))
DATA_DIR = join(BASE_DIR, 'dataset')
FIGURES_DIR = join(BASE_DIR, 'figures')
FIGSIZE = (4, 3)
C1 = '#1F78B4'
C2 = '#FF7F00'


def get_data_file(fn, subfolder=None):
    if subfolder is not None:
        return join(DATA_DIR, subfolder, fn)
    else:
        return join(DATA_DIR, fn)


def get_output_file(fn):
    return join(FIGURES_DIR, fn)


def nplog(a, base=10):
    return (np.log(a) / np.log(base))


def test_in_bot_score(bdf, *dfs):
    s1 = set()
    for df in dfs:
        s1 |= set(df.user_raw_id.tolist())
    s2 = set(bdf.user_raw_id.tolist())
    logger.warning(s1 - s2)
    return s1.issubset(s2)


# Figure 1a in the main paper
def m1a_article_distribution_by_tweets(
        fn1='M1a_ntweets_per_article.lc.csv',
        fn2='M1a_ntweets_per_article.ff.csv',
        ofn='M1a-article-distribution-by-tweets.pdf',
        density=True,
        nbins=15,
        base=2,
        figsize=(4, 3)):
    """
    Caption in the main paper:
    --------------------------
    Figure 1: Online virality of content. (a) Probability distribution
    (density function) of the number of tweets for articles from both
    low-credibility and fact-checking sources. The distributions of the number
    of accounts sharing an article are very similar (see Supplementary 
    Information).

    Input data and instructions:
    ----------------------------
    `fn1` and `fn2` are intermediate results, representing number of tweets
    for each article (low-credibility for `fn1` and fact-checking for `fn2`)
    """
    fn1 = get_data_file(fn1, subfolder='consensus1')
    fn2 = get_data_file(fn2, subfolder='consensus1')
    ofn = get_output_file(ofn)
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    # Note, set max bin points slightly larger than max data value
    # because the calculation may make right most bin points
    # a little smaller than max data value
    bins1 = np.logspace(
        nplog(df1.counts.min(), base),
        nplog(df1.counts.max() + 1, base),
        nbins + 1,
        base=base)
    bins2 = np.logspace(
        nplog(df2.counts.min(), base),
        nplog(df2.counts.max() + 1, base),
        nbins + 1,
        base=base)
    if density is not True:
        w1 = np.ones_like(df1.counts.values) / len(df1.counts.values)
        w2 = np.ones_like(df2.counts.values) / len(df2.counts.values)
    else:
        w1 = None
        w2 = None

    h1, bins1 = np.histogram(
        df1.counts.values, bins=bins1, weights=w1, density=density)
    h2, bins2 = np.histogram(
        df2.counts.values, bins=bins2, weights=w2, density=density)
    f, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('(a) Number of Tweets')
    ax.set_ylabel('PDF')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(
        bins1[:-1], h1, c=C1, marker='o', alpha=0.6, label='Low-credibility')
    ax.plot(bins2[:-1], h2, c=C2, marker='s', alpha=0.6, label='Fact checking')
    ax.legend()
    plt.tight_layout()
    plt.savefig(ofn)


# help function
def prepare_ternary(figsize, scale):
    """Help function to ternary plot"""
    fig, ax = plt.subplots(figsize=figsize)
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    ax.axis('off')
    gm = 0.1 * scale
    blw = 1
    tlw = 1
    # Draw Boundary and Gridlines
    tax.boundary(linewidth=blw)
    tax.gridlines(color='grey', multiple=gm, alpha=0.8)
    # Set Axis labels and Title
    tax.bottom_axis_label(
        r"Retweets $\rightarrow$", offset=-0.08, fontsize='small')
    tax.right_axis_label(r"$\leftarrow$Replies", offset=0.2, fontsize='small')
    tax.left_axis_label(r"$\leftarrow$Tweets", offset=0.2, fontsize='small')

    # ticks
    locations = range(0, scale + 1, 4)
    ticks = ['{}'.format(x * 10) for x in range(0, 11, 2)]
    tax.ticks(
        axis='lbr',
        ticks=ticks,
        locations=locations,
        linewidth=tlw,
        offset=0.03,
        fsize=9,
        clockwise=False)
    return tax


# Figure2a in the main paper
def m2a_types_of_tweets_low_credibility(
        fn='M2a_types_of_tweets.lc.csv',
        ofn='M2a-types-of-tweets-low-credibility.pdf',
        ntop=None,
        least_ntweets=100,
        scale=20,
        style='hexagonal',
        figsize=(5, 3.5)):
    """
    Caption in the main paper:
    --------------------------
    Figure 2: Anomalies. The distribution of types of tweet spreading articles
    from (a) low-credibility ...

    Input data and instructions:
    ----------------------------
    `fn` is a intermediate results, representing the statistics of types of
    tweets for each article. You should have the knowledge of how to recogonize
    different types of tweets.
    """
    fn = get_data_file(fn, subfolder='consensus1')
    ofn = get_output_file(ofn)
    df = pd.read_csv(fn)
    df.set_index('id', inplace=True)
    df['ntotal'] = df.sum(axis=1)
    df = df.sort_values('ntotal', ascending=False)
    logger.info('Total artilces %s', len(df))
    if ntop is not None:
        df = df.iloc[:ntop]
    if least_ntweets is not None:
        df = df.loc[df.ntotal >= least_ntweets]
    logger.info('Number of articles to visualize, %s', len(df))

    X = ((df.nretweets + df.nquotes) / df.ntotal * scale).values
    Y = (df.nreplies / df.ntotal * scale).values
    Z = (df.norigins / df.ntotal * scale).values

    data = heatmap_density(X, Y, Z, scale)
    data = pd.Series(data)
    vmin = data.min()
    vmax = data.max()
    data = data.apply(lambda x: np.log10(x + 1))
    if vmin <= 0:
        vmin = 1
    logger.info('vmin=%s, vmax=%r', vmin, vmax)
    data = data.to_dict()
    cmap = plt.cm.get_cmap('gnuplot2_r')
    tax = prepare_ternary(figsize, scale)
    tax.heatmap(
        data,
        cmap=cmap,
        style=style,
        colorbar=False,
    )
    colorbar_hack(
        ax=tax.ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        log_norm=True,
        cbarlabel='Density')
    # tax.scatter(zip(X, Y), s=49, alpha=0.2)
    plt.tight_layout()
    plt.savefig(ofn)


# Figure 2b in the main paper
def m2b_types_of_tweets_factchecking(
        fn='m2b_types_of_tweets.ff.csv',
        ofn='M2b-types-of-tweets-factchecking.pdf',
        ntop=None,
        least_ntweets=100,
        scale=20,
        style='hexagonal',
        figsize=(5, 3.5)):
    """
    Caption in the main paper:
    --------------------------
    Figure 2: Anomalies. The distribution of types of tweet spreading articles
    from (a) low-credibility ...

    Input data and instructions:
    ----------------------------
    `fn` is a intermediate results, representing the statistics of types of
    tweets for each article. You should have the knowledge of how to recogonize
    different types of tweets.
    """
    fn = get_data_file(fn, subfolder='consensus1')
    ofn = get_output_file(ofn)
    df = pd.read_csv(fn)
    df.set_index('id', inplace=True)
    df['ntotal'] = df.sum(axis=1)
    df = df.sort_values('ntotal', ascending=False)
    logger.info('Total artilces %s', len(df))
    if ntop is not None:
        df = df.iloc[:ntop]
    if least_ntweets is not None:
        df = df.loc[df.ntotal >= least_ntweets]
    logger.info('Number of articles to visualize, %s', len(df))

    X = ((df.nretweets + df.nquotes) / df.ntotal * scale).values
    Y = (df.nreplies / df.ntotal * scale).values
    Z = (df.norigins / df.ntotal * scale).values

    data = heatmap_density(X, Y, Z, scale)
    data = pd.Series(data)
    vmin = data.min()
    vmax = data.max()
    data = data.apply(lambda x: np.log10(x + 1))
    if vmin <= 0:
        vmin = 1
    logger.info('vmin=%s, vmax=%r', vmin, vmax)
    data = data.to_dict()
    cmap = plt.cm.get_cmap('gnuplot2_r')
    tax = prepare_ternary(figsize, scale)
    tax.heatmap(
        data,
        cmap=cmap,
        style=style,
        colorbar=False,
    )
    colorbar_hack(
        ax=tax.ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        log_norm=True,
        cbarlabel='Density')
    # tax.scatter(zip(X, Y), s=49, alpha=0.2)
    plt.tight_layout()
    plt.savefig(ofn)


# Figure 2c in the main text
def m2c_article_gini_distribution_violin(
        fn='M2c_article_gini_coef.csv',
        ofn='M2c-article-gini-distribution-violin.pdf',
        nbins=10,
        base=10,
        figsize=FIGSIZE):
    """
    Caption in the main paper:
    --------------------------
    Figure 2: Anomalies. The distribution of types of tweet spreading articles
    from (a) low-credibility ...

    Input data and instructions:
    ----------------------------
    `fn` is a intermediate results, representing the gini coefficient for each
    low-credibility article.
    """
    fn = get_data_file(fn, subfolder='consensus1')
    ofn = get_output_file(ofn)
    df = pd.read_csv(fn)
    bins = np.logspace(
        nplog(df.ntweets.min(), base),
        nplog(df.ntweets.max() + 1, base),
        nbins + 1,
        base=base)
    # pdb.set_trace()
    data = []
    for i in range(nbins):
        data.append(df.loc[(df.ntweets > bins[i])].gini.values)

    positions = bins[:-1] + np.true_divide(bins[1:] - bins[:-1], 2)
    f, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_ylabel('Gini coefficient')
    ax.set_xlabel('Minimum number of tweets')
    ax.violinplot(
        data,
        positions=positions,
        vert=True,
        widths=1.6 * bins[:-1],
        showmeans=False,
        showextrema=False,
        showmedians=True,
        points=100,
        bw_method='scott',
    )

    ax.set_xscale('log')
    ax.set_xlim([1, 50000])
    ax.set_ylim([-0.05, 0.65])
    plt.tight_layout()
    plt.savefig(ofn)


# Figure 2d in the main paper
def m2d_bots_of_users(fn1='M2d_top1000_users.lc.csv',
                      fn2='M2d_rand1000_users.lc.csv',
                      fn3='user_bot_score.calibrated.csv',
                      ofn='M2d-bots-of-users.pdf',
                      nbins=20,
                      normed=True,
                      figsize=FIGSIZE):
    """
    Caption in the main paper:
    --------------------------
    Figure 2: Anomalies. The distribution of types of tweet spreading articles
    from (a) low-credibility ...

    Input data and instructions:
    ----------------------------
    `fn1`: the top 1000 accounts that post tweets with a link to low-credibility
    sources;
    `fn2`: the random sampled accounts (size=1000) that post tweets with a link
    to low-credibility sources.
    """
    fn1 = get_data_file(fn1, subfolder='consensus1')
    fn2 = get_data_file(fn2, subfolder='consensus1')
    fn3 = get_data_file(fn3, subfolder=None)
    ofn = get_output_file(ofn)
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df3 = pd.read_csv(fn3)
    df3 = df3.loc[df3.bot_score.notnull()]
    s1 = df3.loc[df3.user_raw_id.isin(df1.user_raw_id.values)].bot_score
    s2 = df3.loc[df3.user_raw_id.isin(df2.user_raw_id.values)].bot_score
    a1 = s1.values
    a2 = s2.values
    mu1 = np.mean(a1)
    sigma1 = np.std(a1, ddof=1)
    mu2 = np.mean(a2)
    sigma2 = np.std(a2, ddof=1)
    logger.info('Number of Non-nan values: len(top)=%s, len(rand)=%s', len(a1),
                len(a2))
    logger.info('Ratio of >0.5: r(top)=%s, r(rand)=%s',
                (a1 > 0.5).sum() / float(len(a1)),
                (a2 > 0.5).sum() / float(len(a2)))
    logger.info('mu1=%s, mu2=%s', mu1, mu2)
    logger.info('sigma1=%s, sigma2=%s', sigma1, sigma2)
    logger.info('Welch\'s t-test: %s', ttest_ind(a1, a2, equal_var=False))
    logger.info('Kolmogorov-Smirnov test: %s', ks_2samp(a1, a2))
    logger.info('Mann Whitney U test: %s', mannwhitneyu(a1, a2))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 1, nbins + 1)
    if normed is False:
        w1 = np.ones_like(a1) / len(a1)
        w2 = np.ones_like(a2) / len(a2)
    else:
        w1 = None
        w2 = None
    ax.set_xlim([0, 1])
    ax.hist(
        a1,
        bins,
        weights=w1,
        normed=normed,
        alpha=1,
        label='Most Active',
        histtype='step',
        color=C1)
    ax.hist(
        a2,
        bins,
        weights=w2,
        normed=normed,
        alpha=1,
        label='Random Sample',
        histtype='step',
        color=C2)
    plt.legend(loc='upper right', fontsize='small')
    ax.set_xlabel('Bot Score')
    if normed is True:
        ax.set_ylabel('Probability Density')
    else:
        ax.set_ylabel('$proportion$')
    plt.tight_layout()
    plt.savefig(ofn)


# Figure 3a in main paper
def m3a_bots_in_first_hour_spreading_violin(
        fn1='M3a_first_day_tweets_for_top1000_articles.lc.csv',
        fn2='M3a_sampled_users_with_score.csv',
        ofn='M3a-bots-in-first-hour-spreading-violin.pdf',
        include_theonion=False,
        keep_first_accounts_only=False,
        base=2,
        figsize=FIGSIZE):
    """
    Caption in the main paper:
    --------------------------
    Figure 3: Bot strategies. (a) Early bot support after a viral
    low-credibility article is first shared. We consider a sample of 60,000
    accounts that participate in the spread of the 1,000 most viral stories
    from low-credibility sources. We align the times when each article
    first appears. We focus on a one-hour early spreading phase following
    each of these events, and divide it into logarithmic lag intervals.
    The plot shows the bot score distribution for accounts sharing the articles
    during each of these lag intervals.

    Input data and instructions:
    ----------------------------
    `fn1`: the first day tweets for the top 1000 low-credibility articles.
    `fn2`: the sampled accounts from `fn1` of size=60,000 and the bot score
    is attached.
    """
    fn1 = get_data_file(fn1, subfolder='consensus1')
    fn2 = get_data_file(fn2, subfolder='consensus1')
    ofn = get_output_file(ofn)
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df1 = df1.loc[df1.seconds <= 3600]
    if include_theonion is False:
        # site_id=129 for theonion.com
        df1 = df1.loc[df1.site_id != 129]
    df = pd.merge(df1, df2, on='user_raw_id', how='inner')
    df.sort_values('seconds', inplace=True, ascending=True)
    if keep_first_accounts_only is True:
        logger.info('Keep first appeared tweets only!')
        logger.info('Before drop duplicates: %s', len(df))
        df = df.drop_duplicates(['article_id', 'user_raw_id'], keep='first')
        logger.info('After drop duplicates: %s', len(df))
    X = df.seconds.values + 1
    logger.info('Maximum seconds is: %s', X.max())
    if len(df.loc[df.seconds == 1]) > 0:
        logger.info('bin [1,2) needed!')
        nxbins = int(np.ceil(nplog(X.max(), base=base))) + 1
        xbins = np.ones(nxbins + 1)
        xbins[1:] = 1 + base**np.arange(0, nxbins)
    else:
        logger.info('bin [1,2) not needed!')
        nxbins = int(np.ceil(nplog(X.max(), base=base)))
        xbins = base**np.arange(0, nxbins + 1)
    logger.info('xbins are %s', xbins)
    data = []
    # import pdb; pdb.set_trace()
    for i in range(nxbins - 1):
        dfi = df.loc[(X >= xbins[i]) & (X < xbins[i + 1])]
        data.append(dfi.bot_score.values)
    dfi = df.loc[(X >= xbins[nxbins - 1]) & (X <= xbins[nxbins])]
    data.append(dfi.bot_score.values)
    f, ax = plt.subplots(figsize=figsize)
    positions = xbins[:-1] + np.true_divide(xbins[1:] - xbins[:-1], 2)
    logger.info('positions=%s', positions)
    ax.violinplot(
        data,
        positions=positions,
        vert=True,
        widths=0.6 * xbins[:-1],
        showmeans=False,
        showextrema=False,
        showmedians=True,
        points=100,
    )
    ax.set_xlabel('Lag $+1$ (seconds)')
    ax.set_ylabel('Bot Score')
    ax.set_ylim([-0.05, 0.85])
    ax.set_xscale('log')
    ax.set_xlim([1, 4e3])
    plt.tight_layout()
    plt.savefig(ofn)


# help function for m3b_bots_targeting
def assign_segregated_groups(df, ngroups=3):
    g_size, _ = divmod(len(df), ngroups)
    df['gn'] = -1
    windows = []
    left = 0
    n = len(df)
    df = df.reset_index(drop=True)
    for i in range(ngroups):
        cur_p = (i + 1) * g_size
        right = df.iloc[cur_p - 1].from_bot_score
        windows.append((left, right))
        left = right
    left, right = windows[-1]
    windows[-1] = (left, 1)
    for i, window in enumerate(windows):
        left, right = window
        df.loc[((df.from_bot_score > left) & (df.from_bot_score < right)
                ), 'gn'] = i
    df = df.loc[df.gn != -1]
    return df


def m3b_bots_targeting_3bins(fn='M3b_sampled_mentions.60000.csv',
                             ofn='M3b-bots-targeting-3bins.pdf',
                             nbins=3):
    """
    Caption in the main paper:
    --------------------------
    Figure 3: Bot strategies. (a) Early bot support after a viral
    low-credibility article is first shared. We consider a sample of 60,000
    accounts that participate in the spread of the 1,000 most viral stories
    from low-credibility sources. We align the times when each article
    first appears. We focus on a one-hour early spreading phase following
    each of these events, and divide it into logarithmic lag intervals.
    The plot shows the bot score distribution for accounts sharing the articles
    during each of these lag intervals.

    Input data and instructions:
    ----------------------------
    `fn`: sampled mention tweets that both the mentioning user and mentioned
    user are in file 'M3a_sampled_users_with_score.csv'.
    """
    fn = get_data_file(fn, subfolder='consensus1')
    ofn = get_output_file(ofn)
    df = pd.read_csv(fn)
    df = df.loc[~df.is_via_word_in_tweet_text]
    df = df[['from_raw_id', 'from_bot_score', 'to_followers_count']].copy()
    df = df.groupby('from_raw_id').mean()
    df = df.sort_values('from_bot_score', ascending=True)
    df = df.reset_index(drop=True)
    df = assign_segregated_groups(df, nbins)
    gpb = df.groupby('gn')
    d_mean = gpb.to_followers_count.mean()
    d_std = gpb.to_followers_count.std()
    d_size = gpb.size()
    x = np.array(range(3))
    y1 = d_mean.values / 1e6
    y1err = d_std / d_size.apply(np.sqrt) / 1e6
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(x, y1, yerr=y1err, fmt='--o', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(['Lower Third', 'Middle Third', 'Top Third'])
    ax.set_xlabel('Bot Score Percentile of Mentioning Accounts')
    ax.set_ylabel('Followers of Mentioned Accounts')
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([6.55, 8.15])
    ax.text(
        0,
        1.02,
        r'$\times10^6$',
        transform=ax.transAxes,
        horizontalalignment='left')
    iax = plt.axes([.54, .24, .4, .32])
    df.loc[:, 'to_followers_count'] = np.log10(df.to_followers_count)
    data = []
    for name, gp in df.groupby('gn'):
        data.append(gp['to_followers_count'].values)
    iax.violinplot(
        data,
        positions=x + 0.15,
        vert=True,
        widths=0.8,
        showmeans=False,
        showextrema=False,
        showmedians=True,
        points=100,
    )
    iax.set_xticks(x + 0.15)
    iax.set_xticklabels([])
    iax.set_yticks([0, 4, 8])
    iax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, y: r'$10^%d$' % x))
    plt.tight_layout()
    plt.savefig(ofn)


# Figure 4 in the main paper
def m4_bots_retweeter_and_tweeter(fn1='M4_retweeter_and_tweeter.csv',
                                  fn2='user_bot_score.calibrated.csv',
                                  ofn='M4-bots-retweeter-and-tweeter.pdf',
                                  nbins=20,
                                  figsize=(6, 4.5)):
    """
    Caption in the main paper:
    --------------------------
    Figure 4: Impact of bots on humans. (a) Joint distribution of bot scores
    of accounts that retweeted links to low-credibility articles and accounts
    that had originally posted the links. Color represents the number of
    retweeted messages in each bin, on a log scale. (b) The top projection
    shows the distributions of bot scores for retweeters, who are mostly human.
    (c) The left projection shows the distributions of bot scores for accounts
    retweeted by likely humans (score below 0.5), with a significant portion
    of likely bots.

    Input data and instructions:
    ----------------------------
    `fn1`: retweeter and tweeter from sampled retweets.
    `fn2`: bot score for retweeter and tweeter.
    """
    fn1 = get_data_file(fn1, subfolder='consensus1')
    fn2 = get_data_file(fn2, subfolder=None)
    ofn = get_output_file(ofn)
    df1 = pd.read_csv(fn1)
    df2 = pd.read_csv(fn2)
    df2 = df2.loc[df2.bot_score.notnull()]
    df = pd.merge(
        df2, df1, left_on='user_raw_id', right_on='retweeter_id', how='inner')
    df.rename(columns=dict(bot_score='retweeter_score'), inplace=True)
    df = pd.merge(
        df2, df, left_on='user_raw_id', right_on='tweeter_id', how='inner')
    df.rename(columns=dict(bot_score='tweeter_score'), inplace=True)
    df = df[['retweeter_score', 'tweeter_score']].copy()
    x = df.retweeter_score.values
    y = df.tweeter_score.values
    H, xedges, yedges = np.histogram2d(
        x, y, bins=np.linspace(0.0, 1.0, nbins + 1))
    # X1, Y1, horizontal curve
    df = pd.DataFrame(H)
    X1 = xedges[:-1] + np.true_divide(xedges[1:] - xedges[:-1], 2)
    Y1 = df.sum(axis=1) / df.sum(axis=1).sum()
    # X2, Y2, vertical curve
    df2 = df.iloc[:10]
    X2 = df2.sum(axis=0) / df.sum(axis=0).sum()
    Y2 = X1
    # X3, Y3, heatmap
    X3, Y3 = np.meshgrid(xedges, yedges)

    plt.figure(figsize=figsize)
    # Display -> Figure
    # figtr = fig.transFigure.inverted()
    gs = gridspec.GridSpec(
        2,
        4,
        wspace=0.3,
        hspace=0.2,
        width_ratios=[2.5, 6, 0.2, 0.5],
        height_ratios=[0.8, 2])
    # gs[0, 1].update(hspace=0.1)
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    ax4 = plt.subplot(gs[1, 3])
    # horizontal curve
    ax1.plot(X1, Y1)
    # verticla curve
    ax2.plot(X2, Y2)
    # heatmap
    im = ax3.pcolormesh(
        X3,
        Y3,
        H.T,
        norm=mpl.colors.LogNorm(vmin=H.min() + 1, vmax=H.max()),
        cmap='gnuplot2_r')
    # color bar
    cb = plt.colorbar(im, ax=ax4, orientation='vertical', fraction=0.9)
    cb.set_label('Retweets')
    # disable xticklabels of ax1
    ax1.set_xticklabels([])
    # disable xticklabels of ax2
    ax2.set_yticklabels([])
    # set tick on right yaxis of ax2
    ax2.yaxis.tick_right()
    # add vertical dashed line on ax1
    ax1.axvline(0.5, color='k', linestyle='--', lw=0.5)
    # add xlabel
    ax1.set_ylabel('$\Pr(x)$')
    ax1.set_ylim((0, 0.2))
    # add horizontal dashed line on ax2
    ax2.axhline(0.5, color='k', linestyle='--', lw=0.5)
    # set inverse xlim of ax2, making xaxis increasing on inverse direction
    ax2.set_xlim((0.1, 0))
    # add ylabel
    ax2.set_xlabel('$\Pr(y|x\leq 0.5)$')
    #fontsize=rcParams['axes.titlesize'],
    #fontweight=rcParams['axes.titleweight'])
    # set xlabel of ax3
    ax3.set_xlabel('Bot Score of Retweeter, $x$')
    # set ylabel of ax3
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('Bot Score of Tweeter, $y$')
    # disable axis
    ax4.axis('off')
    # add rectangel for ax3
    rect = mpl.patches.Rectangle(
        (0.015, 0.008),
        0.49,
        0.98,
        alpha=1,
        ec='black',
        facecolor=None,
        fill=False,
        ls='-.',
        lw=1.5)
    ax3.add_patch(rect)
    plt.savefig(ofn)


# help function for m5_bots_robustness
def remaining_claim_url(tweet_df, user_centrality_series):
    tweet_df = tweet_df.copy()
    ordered_uid = [
        str(uid)
        for uid in user_centrality_series.sort_values(ascending=False).index
    ]
    user_rank_dict = {uid: rank + 1 for rank, uid in enumerate(ordered_uid)}
    retweet_removal_order = \
        np.minimum(
            tweet_df.from_raw_id.apply(lambda uid: user_rank_dict[uid]),
            tweet_df.to_raw_id.apply(lambda uid: user_rank_dict[uid])
        )
    tweet_df['retweet_removal_order'] = retweet_removal_order
    link_removal_order = tweet_df.groupby('article_id')[
        'retweet_removal_order'].max()
    counts = {user_rank_dict[uid]: 0.0 for uid in ordered_uid}
    counts.update(link_removal_order.value_counts().to_dict())
    removed = pd.Series(counts).sort_index()
    remaining = 1.0 - removed.cumsum() / removed.sum()
    remaining = pd.concat([pd.Series([1.0], index=[0.0]), remaining])
    remaining.index += 1
    return remaining


# help function for m5_bots_robustness
def remaining_percentage(tweet_df, user_centrality_series):
    ordered_uid = [
        str(uid)
        for uid in user_centrality_series.sort_values(ascending=False).index
    ]
    user_rank_dict = {uid: rank + 1 for rank, uid in enumerate(ordered_uid)}
    removal_order = \
        np.minimum(
            tweet_df.from_raw_id.apply(lambda uid: user_rank_dict[uid]),
            tweet_df.to_raw_id.apply(lambda uid: user_rank_dict[uid])
        )
    counts = {user_rank_dict[uid]: 0.0 for uid in ordered_uid}
    counts.update(removal_order.value_counts().to_dict())
    removed = pd.Series(counts).sort_index()
    remaining = 1.0 - removed.cumsum() / removed.sum()
    remaining = pd.concat([pd.Series([1.0], index=[0.0]), remaining])
    remaining.index += 1
    return remaining


def m5_bots_robustness(fn1='M5_retweet.20171108.lc.csv',
                       fn2='M5_centralities.csv',
                       output='M5-bots-robustness.pdf'):
    """
    Caption in the main paper:
    --------------------------

    Input data and instructions:
    ----------------------------
    `fn1`: retweets with links to low-credibility sources before 2017-11-08.
    `fn2`: three centralities for users with large k-core number: in-strengh,
    out-strengh, and bot-score.
    """
    fn1 = get_data_file(fn1, subfolder='consensus1')
    fn2 = get_data_file(fn2, subfolder='consensus1')
    output = get_output_file(output)

    df1 = pd.read_csv(fn1, dtype=dict(from_raw_id=str, to_raw_id=str))
    df2 = pd.read_csv(fn2, index_col=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
    plot_kw = dict(ax=ax1, logx=True)
    remaining_claim_url(df1, df2['in_s']).plot(
        label='Activity', linestyle='-.', **plot_kw)
    remaining_claim_url(df1, df2['out_s']).plot(
        label='Influence', linestyle='--', **plot_kw)
    remaining_claim_url(df1, df2['bot_score']).plot(
        label='Bot Score', linestyle=':', **plot_kw)
    ax1.set_xlabel('number of spreaders disconnected')
    ax1.set_ylabel('fraction of claim links remaining')
    ax1.legend()

    plot_kw = dict(ax=ax2, logx=True)
    remaining_percentage(df1, df2['in_s']).plot(
        label='Activity', linestyle='-.', **plot_kw)
    remaining_percentage(df1, df2['out_s']).plot(
        label='Influence', linestyle='--', **plot_kw)
    remaining_percentage(df1, df2['bot_score']).plot(
        label='Bot Score', linestyle=':', **plot_kw)
    ax2.set_xlabel('number of spreaders disconnected')
    ax2.set_ylabel('fraction of claim links remaining')
    # ax2.legend(loc=3, fontsize=9)
    ax2.set_xlabel('number of spreaders disconnected')
    ax2.set_ylabel('fraction of claim posts remaining')
    plt.tight_layout()
    plt.savefig(output)


# Figure 6 in the main paper
def m6_sources_popularity(fn='M6_top_sources_popularity.csv',
                          ofn='M6-top-sources-popularity.pdf',
                          figsize=(12, 5)):
    fn = get_data_file(fn, subfolder='consensus1')
    ofn = get_output_file(ofn)
    df = pd.read_csv(fn)
    df.rename(
        columns=dict(
            median_of_tweets_per_article='at_medians',
            median_of_bot_score_of_top_users='bs_medians',
            total_tweets='ntweets'),
        inplace=True)
    df = df.sort_values('ntweets', ascending=False)
    # trim domains top label
    df.domain = df.domain.apply(lambda x: x[:-4])
    # size configuration
    base_v = df.at_medians.min()
    scale = np.sqrt(df.at_medians.values / base_v)
    base_s = 180
    sizes = scale * base_s
    # color configuration
    color_low_credibility = 'red'
    color_satire = 'orange'
    color_fact_checking = 'deepskyblue'
    df['color'] = color_low_credibility
    df.loc[df.site_type == 'fact_checking', 'color'] = color_fact_checking
    df.loc[df.site_type == 'satire', 'color'] = color_satire
    # start ploting
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        df.ntweets.values,
        df.bs_medians.values,
        s=sizes,
        color=df.color.values,
        edgecolors='k',
        alpha=0.8)
    # Set lables for the scattered point
    for txt, x, y, sc in zip(
            range(len(df)), df.ntweets.tolist(), df.bs_medians.tolist(),
            scale):
        ax.annotate(
            txt + 1,
            xy=(x, y),
            # xytext=(0, sc * base_yt),
            # textcoords='offset points',
            # fontsize=nplog(sc + 1, 2) * base_fs,
            ha='center',
            va='center')
    # set legend
    handlers = [
        mpl.lines.Line2D(
            [0], [0],
            linestyle='none',
            mfc='black',
            mec='none',
            marker=r'$\mathregular{{{}.}}$'.format(i),
            markersize=7.5) for i in range(1, 10)
    ]
    handlers += [
        mpl.lines.Line2D(
            [0], [0],
            linestyle='none',
            mfc='black',
            mec='none',
            marker=r'$\mathregular{{{}.}}$'.format(i),
            markersize=12.5) for i in range(10,
                                            len(df) + 1)
    ]
    plt.legend(handlers, df.domain.tolist(), ncol=3, loc=2)
    # set colors of legend text
    # pdb.set_trace()
    for lh, c in zip(ax.get_legend().get_texts(), df.color.tolist()):
        lh.set_color(c)
    ax.set_xlabel('Total Tweet Volume', fontsize=14)
    ax.set_ylabel('Median Bot Score of Active Accounts', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_xscale('log')
    ax.set_xlim(1e4, 1e7)
    plt.tight_layout()
    plt.savefig(ofn)
