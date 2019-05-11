#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:03:36 2019

@team: A team has no name
@author: msarjun, gizmowiki, and vidhyasri
"""
# library imports
import re
import sys
import random

# Eliza welcome message and asking for name
## e.g. inputs: Adam, My name is Adam, I don't want to tell you my name, etc.
## bye, Bye, quit, Quit, exit, Exit to quit Eliza ...
print("To exit please enter any of these commands-[Bye,bye,Exit,exit,Quit,quit]")
text= input("Eliza: "+"Hiii! I am Eliza, the chatbot. What is your name?\nYou: ")
user_name = "annonymous"
if re.findall('(.*) ?([Bb]ye|[Qq]uit|[Ee]xit) ?(.*)',text):
	print("Eliza: "+"Thanks for talking with me.Have a great day!")
	sys.exit()
elif re.findall(r"""(.*) name is (\w+)\b ?(.*)""", text):
	user_name = re.findall(r"""(.*) name is (\w+)\b ?(.*)""", text)[0][1]
	print("Eliza: "+"Nice to meet you", user_name)
elif re.findall('^([A-Z]?[a-z]+)$',text):# re.findall('([A-Z]?[a-z]+)$',text):
	user_name=re.findall('([A-Z]?[a-z]+)$',text)[0]
	print("Eliza: "+"Great to meet you {0}".format(user_name))
else:
	print("Eliza: "+"I understand that you want to be anonymous")

# Initializing emotions (Adjective -> Noun)
emotion={"hungry":"hunger","angry":"anger","sad":"sadness","happy":"happiness","excited":"excitement",
			"anxious":"anxiety","nervous":"nervousness","afraid":"fear","joyful":"joy","thirsty":"thirst",
			"stupid":"stupidity","innocent":"innocence","crave":"craving", "depress": "depression", "lonely": "loneliness"}

# Initializing response structure w.r.t. requested text
## _str_ token replaced by certain emotions to target and _name_ is replaced by the user's name for more personalized experience
response={
			 r'(.*) ?I need (.*)': ["why do you need _str_?","Do you need _str_?","Are you sure you need _str_?"],
			 r'(.*) ?[bB]ecause (.*)': ['Is that the real reason _str_','Will this make you happy _str_'],
			 r'(.*) ?I want (.*)': ["Why do you want _str_?","Do you want _str_?","Are you sure you want _str_?"],
			 r'(.*) ?I think (I|about) (.*)': ["Why do you think about your _str_?", "What makes you think about your _str_", "Why don't you tell me more about your _str_?"],
			 r'(.*) ?I am (.*)': ["_name_, Tell me about your _str_.","Pleae talk about your _str_.","_name_, Lets talk about your _str_."],
			 r'(.*) ?I dont (.*)': ["Why you dont _str_?","You should _str_.","You will feel better if you _str_."],
			 r'(.*) ?[yY]ou are (.*)': ["Why do you think I am _str_?","What makes you think I am _str_?","Does it makes you happy that I am _str_?"],
			 r'(.*) ?[Ss]orry ?(.*)': ["Umm, don't be sorry, _name_. It's all right...","Never mind _name_","It's all right _name_?"],
			 r'(.*) ?I (\w+) you ?(.*)': ["What makes you _str_ me?", "_name_, Did you just said that you _str_ me?"],
			 r'(.*) ?you (.*)': ["Are you talking about me. Let’s talk about you.", "This is about you, _name_!"],
			 r"""(.*) ?[Mm]aybe (.*)""": ["Why the uncertain tone, _name_?", "Be specific, _name_."],
			 r'(.*) ?[pP]roblem (.*)': ["What is your problem _name_. Can you explain more on that?", "Is that problem bothering you _name_?"],
			 r'(.*) ?([dD]reams?|dreamt) ?(.*)': ["_name_, Can you explain me in detail about your dream?"],
			 r'(I)(.*)(like|kill|love|dance|hate)(.*)(you)':["Perhaps in your fantasies", "Perhaps in your dreams", "Perhaps you wish"],
			 r'(.*) ?(mom|dad|mother|father|boyfriend|girlfriend|son|daughter|husband|wife|friend|children).*':["_name_, Tell me more about _str_", "Do you like your _str_?"],
			 r'.*(not sure|may|may not|maybe|maybe not).*':["_name_, Why is there an uncertainty in your tone?"],
			 r'[0-9]+':["_name_, I’m afraid, I can't decode numbers"]
			 }


isRepeatCounter = 1 # If user repeats more than 3 times, Eliza responds appropriately
old_text = ""
text=input("Eliza: "+"How can I help you today?\nYou: ")
while not re.findall('(.*) ?([Bb]ye|[Qq]uit|[Ee]xit) ?(.*)',text):
	if(text == old_text):
		isRepeatCounter += 1
		if isRepeatCounter >= 3: # checking if counter reaches 3
			isRepeatCounter = 1
			print("Why are you repeating?")
			text = input()
			continue
	old_text = text # to check repetition
	# [accepted_pattern_text] --- For more than one request regex match, store it in a list and pick randomly
	accepted_pattern_text = []
	for pattern in response.keys():
		if re.match(pattern,text):
			accepted_pattern_text.append(pattern)
	if accepted_pattern_text:
		pattern = random.choice(accepted_pattern_text) # Chooses one random pattern from accepted patterns list (accepted_pattern_text)
		match = response[pattern] # stores the list of possible responses
		relevant_group=re.findall(pattern,text)[0][-1] # Store the matched string followed in the request regex, which might contain the emotion
		selected_emotion = [relevant_group.replace(x, y) for x, y in emotion.items() if x in relevant_group] # All the emotions present in the relevant_group
		if selected_emotion: # If there was at least one emotion from our list present in the relevant_group, then change the emotion verb to its noun
			print("Eliza: "+random.choice(match).replace("_str_", random.choice(selected_emotion)).replace('_name_', user_name)) # choose a random response from the possible responses
		else: # If not present just repeat the relevant_group
			print("Eliza: "+random.choice(match).replace("_str_", relevant_group).replace('_name_', user_name)) # choose a random response from the possible responses
	else:
		print("Eliza: "+"Hmm, I didn't quite understand, Can you tell me more about it in a different way!") # Default answer for unmatched patters Eliza didn't understood

	text=input("You: ") # Ask for next input, till user says [qQ]uit, [bBye], ...
print("Thank you for talking to me . I hope to see you again") # Thank you ending note
