# List of the basic nature options, each number represents a prior value of each function that can be used
# as a basic nature of a user in the simulation
# if the number is 0, the function will never be used
# if the number is 1 or more, the prior probability of that function will be multiplied by that number
# and then normalized (sum to 1)

# The basic nature options are:
# Oracle is always used, so it is not included in the list
# 0idx: random action
# 1idx: action based on the history of the bot and the review quality
# 2idx: topic based action
# 3idx: static LLM based action
# 4idx: dynamic LLM based action

pers = [[1, 1, 1, 0, 0],
[1, 3, 1, 0, 0],
[1, 1, 3, 0, 0],
[1, 3, 3, 0, 0],
[0, 0, 1, 0, 0],
[0, 1, 0, 0, 0],
[0, 1, 1, 0, 0],
[1, 0, 0, 0, 0],
[1, 0, 1, 0, 0],
[1, 1, 0, 0, 0],
[0, 0, 0, 1, 0],
[0, 0, 0, 0, 1],
[1, 1, 0, 1, 0],
[1, 1, 0, 0, 1],
[0, 1, 0, 1, 0],
[1, 0, 0, 0, 0],
[1, 0, 0, 1, 0],
        ]