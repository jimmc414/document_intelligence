# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
import pyperclip # import pyperclip module
openai.api_key = os.getenv("OPENAI_API_KEY")

print("Creating a chat completion object with the specified messages and model...") # print console output
chat_completion = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are an expert python programmer and AI Document Intelligence implementation specialist."},
    {"role": "user", "content": "Build a program with these requirements: Must create an example implementation and instructions for a script that uses NER to assign a cross reference number that corresponds with a remote system.  The program will use a csv to cross reference to account numbers in the remote system. "}
  ]
)

# Print and save the generated message content
output = chat_completion.choices[0].message.content # assign output to a variable
print("The chat completion object has been created successfully.") # print console output
print("The generated message content is:") # print console output
print(output) # print output to screen
with open("gpt4_output.txt", "w") as f: # open file object in write mode
  f.write(output) # write output to file
  print("The output has been saved to gpt4_output.txt file.") # print console output
pyperclip.copy(output) # copy output to clipboard
print("The output has been copied to the clipboard.") # print console output

# Save the chat history to another file
with open("gpt4_history.txt", "a") as h: # open file object in append mode
  h.write("\n".join([m["content"] for m in chat_completion.messages])) # write messages to file
  h.write("\n" + output + "\n") # write output to file
  print("The chat history has been saved to gpt4_history.txt file.") # print console output
