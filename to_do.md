# Work to Do
!Before doing anything. Make yourself familiar with all the details in this project to avoid further mistakes.

## Collect Different Data
The assignment requirements said:For the existing feature extraction methods, collect training data, try more than three supervised learning methods, write an introduction to learning methods, and report the performance comparison.     
So follow these steps:
- For each level of the game. Use train_rl_agent.py to train specific agent for that level. Recommended --total-timesteps=2500000
- For each level, run rl_play.py with the agent for this level to collect the winning game data the agent plays. For each level, set wins=80, --deterministic, level to be the corresponding level.
- In directory logs, there maybe empty directories due to some unexplainable bug in rl_play.py Just delete the empty directories.


## Train Different Agents
Regarding the assignment requirement above, you should:    
- Keep the current learn.py unmodified. And analogously, create another two learn.py(you decide the names) with different learning methods, which though, **MUST** be supervised learning methods.
- Run the three learn.py just created respectively. For each type of agent and each level, run learn.py once(so totally 3*6=18 times). Note we count 'all', for reference of which you can check plugins.py, as one separate level. You can modify the models' saving paths' patterns, like appending the methods' name, to tell the difference.
- Design a script which tests 18 agents we just created in a loop and save scores for analysis. For leveled agent, test the agent with its corresponding game level. For 'all' agent, test the agent with all levels(0~4) iteratively. So you need to run 15\*1+5\*3=30 rounds in total. In each round, run 30 tests. So you need to run totally 30*30=900 tests. Save all the tests' final scores in a json file, with information of agent type and game level about each score.
- Design a script to create tables and statistic figures to describe the scores we collected in the previous step. Use matplot to draw figures. Show differences of agent types and levels in the tables and figures, and create a new directory to save the statistic tables and figures.

## Improve the Feature Extraction Methods
The assignment requirements also said:Try to modify the feature extraction method to get better learning performance.   
So do the following:
- Improve the feature extraction method according to your will.
- Choose one type of agent(the best supervised learning method above in your opinion) and repeat the research procedures as we did above. Don't forget to collect scores.
- Compare the new feature extraction method's scores with the old one's scores. Save the statistic tables and figures as well.

# HARD RULES
- There are some remaining models and logs of previous research data. You need to delete the redundant files in figs, logs, and models to begin a fresh new experiment based entirely on the instructions I gave you.
- Don't modify env.py, play.py, test.py.
- Try to be comprehensive when analyzing the scores through tables and figures.
- Make sure that the things you do is correct.
- You can make modifications if I failed to cover them as long as you think they're reasonable, like changing the saving-path-names' patterns.
- Don't be hard requirements.txt You're allowed to modify it.