from sklearn.preprocessing import LabelEncoder
from statistics import stdev, harmonic_mean
import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def compute_user_statistics(response):
    """
    Given a set of tweets calculate the user statistics.

    :param response: response of twitter api statuses.user_timeline.
    :return:
    """

    # TODO: Finish function
    # favorites count               DONE
    # verified                      DONE
    # plain statuses                DONE
    # replies received              PROBLEM how to get replies received
    # replies given                 DONE
    # retweets                      DONE
    # mentions                      DONE
    # total URL                     DONE
    # total hashtags                DONE
    # promotions score              DONE (check edit distance)
    # life time                     DONE (improve get_time_difference)
    # tweet spread/influence        PROBLEM how to get total time taken for first 100 retweets
    # std urls                      DONE
    # std hashtags                  DONE
    # user collective activeness    DONE (improve get_collective_activeness)
    # degree of inclination         DONE
    # collective influence          DONE

    # OTHER FEATURES (commented)
    # description                   DONE
    # tweets                        DONE
    # has_description               DONE
    # time_stamp                    PENDING time between now (or last tweet) and third last tweet

    row = {}

    hashtag_count = 0
    url_count = 0
    plain_statuses = 0
    mentions_count = 0
    retweet_count = 0
    replies_given_count = 0
    personal_tweets = 0
    hashtag_list = []
    url_list = []
    tweets = []
    n_tweet = 0
    t1 = 0
    t2 = 0

    for el in response:
        n_tweet += 1
        if n_tweet == 1:
            t1 = el['created_at']
        elif n_tweet == 3:
            t2 = el['created_at']
        hashtags = len(el['entities']['hashtags'])
        user_mentions = len(el['entities']['user_mentions'])
        urls = len(el['entities']['urls'])
        if hashtags + user_mentions + urls == 0:
            plain_statuses += 1
            tweets.append(el['full_text'])
        if 'retweeted_status' in el.keys():
            retweet_count += 1
        else:
            personal_tweets += 1
        if el['is_quote_status']:
            replies_given_count += 1
        hashtag_count += hashtags
        mentions_count += user_mentions
        url_count += urls
        hashtag_list.append(hashtags)
        url_list.append(urls)

    time_stamp = get_time_difference(t2, t1, unit='days')

    # DONE FEATURES
    row['plain_statuses'] = plain_statuses  # represents the #plain statuses posted without #, URLs or mentions.
    row['mentions'] = mentions_count  # represents the number of mentions found in the tweets posted by a user.
    row['total_url'] = url_count  # represents total number of URLs used in tweets.
    row['total_hashtag'] = hashtag_count  # represents total number of hashtags used in tweets.
    row['std_url'] = stdev(url_list)  # represents the stdev of URLs that a user embedded in his tweets.
    row['std_hashtags'] = stdev(hashtag_list)  # represents the stdev of hashtags that a user used in his tweets.
    row['retweets'] = retweet_count  # represents the number of retweets posted by a user using other users' tweets.
    row['replies_given'] = replies_given_count  # specifies the number of replies given on other users' tweets.
    row['degree_inclination'] = get_degree_inclination(personal_tweets, retweet_count)

    # PENDING FEATURES
    # row['replies_received'] = None  # represents the number of statuses that received replies from other users.

    # EXTRA FEATURES
    # row['text'] = tweets

    return row, time_stamp


    def edit_distance(s1, s2):
    """
    Implementation of Levenshtein distance (https://en.wikipedia.org/wiki/Levenshtein_distance)
    :param s1:
    :param s2:
    :return: levenshtein distance
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_

    return distances[-1]
