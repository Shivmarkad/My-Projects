import random

count_rock=0
count_paper=0
count_scissors=0

def update_counts(user_input):

  global count_paper, count_rock, count_scissors

  if user_input == 0:

    count_rock += 1

  elif user_input == 1:

    count_paper += 1

  else:

    count_scissors += 1
    
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

player_score=0
comp_score=0
print("\nPlacing the newline character before the string.")
print("Placing the newline character before the string.\n")

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

valid_entries = ['0', '1', '2']

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


