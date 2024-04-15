import numpy as np
import json

################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2


################################
# STRATEGIES
################################

def correct_action(information):
    """
    Oracle strategy, always returns the correct action.
    @param information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0


def random_action(information):
    """
    Random strategy, returns a random action.
    @param information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    return np.random.randint(2)


def user_rational_action(information):
    """
    Rational user strategy, returns the action that maximizes the expected utility (based on the review provided).
    Returns 1 if the review is higher than 8, 0 otherwise.
    @param information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    """
    Like the rational user strategy, but only returns 1 if the review is higher than 9.
    @param information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    """
    Like the rational user strategy, but only returns 1 if the review is higher than 7.
    @param information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    """
    TIT-for-TAT strategy that looks one step back.
    @param information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0

def user_short_topic_t4t(positive_topics, negative_topics, quality_threshold):
    """
    Like the hard user t4t, but the quality of the review is not based on the godly information, but on the topics
    in the actual review.
    @param positive_topics: List of topics that are considered positive
    @param negative_topics: List of topics that are considered negative
    @param quality_threshold: The threshold for the review to be considered good
    @return: A function(information) that based on the given information and the pre-made topics,
    returns the action.
    """

    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"][topic]) * 2 / (rank + 1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"][topic]) * 2 / (rank + 1)

        if len(information["previous_rounds"]) == 0 \
                or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                    information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
                or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                    information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
            if information["bot_message"] >= quality_threshold:  # good hotel
                return 1
            else:
                return 0
        else:
            return 0

    return func


def user_picky_short_t4t(information):
    """
    Like TIT-for-TAT, but agrees to go to the hotel if its current review is higher than 9.
    @param information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if (information["bot_message"] >= 9
            or ((information["bot_message"] >= 8)
                and (len(information["previous_rounds"]) == 0
                     or (information["previous_rounds"][-1][REVIEWS].mean() >= 8)))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    """
    Like TIT-for-TAT, but only agrees if the agent has never lied. (Trigger strategy.)
    @param: information: The information of the current round.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0

def user_hard_window_t4t(information, history_window):
    """
    Like hard TIT-for-TAT, but only looks in a certain history window.
    @param: information: The information of the current round.
    @param: history_window: the window of relevant information (how far back the user remembers).
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"][-history_window:]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0

def user_hard_topic_t4t(positive_topics, negative_topics, quality_threshold):
    """
    Like the hard user t4t, but the quality of the review is not based on the godly information, but on the topics
    in the actual review.
    @param positive_topics: List of topics that are considered positive
    @param negative_topics: List of topics that are considered negative
    @param quality_threshold: The threshold for the review to be considered good
    @return: A function(information) that based on the given information and the pre-made topics,
    returns the action.
    """

    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"][topic]) * 2 / (rank + 1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"][topic]) * 2 / (rank + 1)

        if len(information["previous_rounds"]) == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                         or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                        information["previous_rounds"]])) == 1:  # cooperation
            if information["bot_message"] >= quality_threshold:  # good hotel from the users perspective
                return 1
            else:
                return 0
        else:
            return 0

    return func

def user_soft_t4t(information, threshold):
    """
    Like hard TIT-for-TAT, but only agrees if the number of lies is below a given threshold.
    @param: information: The information of the current round.
    @param: threshold: the amount of lying that is allowed.
    @return: 1 if the action is to go to the hotel, 0 otherwise.
    """
    if len(information["previous_rounds"]) == 0 \
            or np.average(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) > threshold:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0

def user_soft_window_t4t(information, threshold, history_window):
    """
        Like soft TIT-for-TAT, but only looks at the history in a certain window.
        @param: information: The information of the current round.
        @param: threshold: the amount of lying that is allowed.
        @param: history_window: the window of relevant information (how far back the user remembers).
        @return: 1 if the action is to go to the hotel, 0 otherwise.
        """
    if len(information["previous_rounds"]) == 0 \
            or np.average(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][-history_window:]])) > threshold:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_soft_topic_t4t(positive_topics, negative_topics, quality_threshold):
    """
    Like the soft user t4t, but the quality of the review is not based on the godly information, but on the topics
    in the actual review.
    @param positive_topics: List of topics that are considered positive
    @param negative_topics: List of topics that are considered negative
    @param quality_threshold: The threshold for the review to be considered good
    @return: A function(information, threshold) that based on the given information and the pre-made topics,
    returns the action.
    """
    def func(information, threshold):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"][topic]) * 2 / (rank + 1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"][topic]) * 2 / (rank + 1)

        if len(information["previous_rounds"]) == 0 \
                or np.average(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                         or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                        information["previous_rounds"]])) > threshold:  # cooperation
            if information["bot_message"] >= quality_threshold:  # good hotel from the users perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func

def user_soft_topic_window_t4t(positive_topics, negative_topics, quality_threshold, history_window):
    """
    Like the soft user topic t4t, but only looks in a certain history window.
    @param positive_topics: List of topics that are considered positive
    @param negative_topics: List of topics that are considered negative
    @param quality_threshold: The threshold for the review to be considered good
    @param history_window: the window of relevant information (how far back the user remembers).
    @return: A function(information, threshold) that based on the given information and the pre-made topics,
    returns the action.
    """
    def func(information, threshold):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"][topic]) * 2 / (rank + 1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"][topic]) * 2 / (rank + 1)

        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.average(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                         or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                        information["previous_rounds"][-history_window:]])) > threshold:  # cooperation
            if information["bot_message"] >= quality_threshold:  # good hotel from the users perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func


def history_and_review_quality(history_window, quality_threshold):
    """
    Like hard TIT-for-TAT, but only looks in a certain history window.
    (Similar to a person forgetting what was previously)
    @param: history_window: the window of relevant information (how far back the user remembers).
    @param: quality_threshold: what counts as a good hotel from the perspective of the user.
    @return: function(information) which returns 1 or 0 based on the history and the quality threshold.
    """
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][-history_window:]])) == 1:
            # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0
    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    """
    Topic based strategy, returns the action based on the topics in the review.
    @param positive_topics: List of topics that are considered positive
    @param negative_topics: List of topics that are considered negative
    @param quality_threshold: The threshold for the review to be considered good
    @return: A function that based on the given information and the pre-made topics, returns the action
    """
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"][topic])*2/(rank+1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"][topic])*2/(rank+1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0
    return func


def LLM_based(is_stochastic):
    """
    LLM based strategy, returns the action based on the LLM score.
    @param is_stochastic: If False, returns 1 with probability of the LLM score. Otherwise, returns 1 with probability of 0.5.
    """
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)
        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)
        return func
