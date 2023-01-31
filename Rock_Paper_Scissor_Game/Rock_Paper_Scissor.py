# Importing random module to generate random integer
import random


# Initially assigning zero to count veriables  
count_rock=0
count_paper=0
count_scissors=0


# Creating a update_counts function
def update_counts(user_input):

  global count_paper, count_rock, count_scissors

  if user_input == 0:

    count_rock += 1

  elif user_input == 1:

    count_paper += 1

  else:

    count_scissors += 1

# Creating predict function to predict the users input 
def predict():

  if count_rock > count_paper and count_rock > count_scissors:

    pred = 1

  elif count_paper > count_rock and count_paper > count_scissors:

    pred = 2

  elif count_scissors > count_rock and count_scissors > count_paper:

    pred = 0

  else:
    
    pred = random.randint(0, 2)

  return pred

# Initially assigning zero to player_score and computer score veriables
player_score=0
comp_score=0

#Creating update score funtion to update the score of computer and user
def update_scores(user_input):

  global player_score, comp_score

  pred = predict()

  if user_input == 0:

    if pred == 0:

      print("\nYou played ROCK(0), computer played ROCK.")
      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)

    elif pred == 1:

      print("\nYou played ROCK(0), computer played PAPER.")

      comp_score += 1

      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)

    else:

      print("\nYou played ROCK(0), computer played SCISSORS.")
      
      player_score += 1

      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)

  elif user_input == 1:

    if pred == 1:

      print("\nYou played PAPER(1), computer played PAPER.")
      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)

    elif pred == 0:

      print("\nYou played PAPER(1), computer played ROCK.")

      player_score += 1

      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)


    else:

      print("\nYou played PAPER(1), computer played SCISSORS.")

      comp_score += 1

      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)

  else:

    if pred==2:

      print("\nYou played SCISSORS(2), computer played SCISSORS")
      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)

    elif pred==0:

      print("\nYou played SCISSORS(2), computer played ROCK.")

      comp_score += 1

      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)

    else:

      print("\nYou played SCISSORS(2), computer played PAPER.")

      player_score += 1

      print("\nComputer Score: ", comp_score, "\nYour Score: ", player_score)
      
# Valid entries that player has to enter 

valid_entries = ['0', '1', '2']

# Running while loop for infinite time 
while True:

  user_input = input("Enter 0 for ROCK, 1 for PAPER and 2 for SCISSORS: ")

  print("You Entered the Number : ",user_input)

  while user_input not in valid_entries:

    print("\nInvalid Input!")

    user_input = input("Enter 0 for ROCK, 1 for PAPER and 2 for SCISSORS: ")

    print("You Entered the Number : ",user_input)

  user_input = int(user_input)

  update_scores(user_input)

  update_counts(user_input)

  if player_score==3:

    print("You won!")
    break

  elif comp_score == 3:

    print("Computer Won!")
    break


