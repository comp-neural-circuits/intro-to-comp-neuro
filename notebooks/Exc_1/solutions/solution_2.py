'''
 Excercicse 2: We now want to combine all the learned concepts into a small task
 
 1) Implement a for loop that goes from 0 to 50
 2) within this for loop we want to print the number for all values that are divisible by 10
 3) When the loop is finished, print how often we printed a number
 
 '''

counter = 0 # we initialize our counter variable
for ii in range(51): 
# it is important to go to 51, because the the range is exclusive on the upper bound
    if ii > 0 and ii%10 == 0: 
        print (ii)
        counter += 1 # this is equivalent to counter = counter + 1

print (counter)


''' Alternative solution '''
list_of_printed_numbers = []
for ii in range(51):
    if ii > 0 and ii%10 == 0:
        print (ii)
        list_of_printed_numbers.append(ii)

print (len(list_of_printed_numbers))