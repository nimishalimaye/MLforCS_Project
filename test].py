import grammar_check
tool = grammar_check.LanguageTool('en-GB')
text = 'hi my name is kabab'

matches = tool.check(text)
print(len(matches))