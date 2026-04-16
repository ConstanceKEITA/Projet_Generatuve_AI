import re

def calculate(expression: str) -> str:
    try:
        expression = (expression
                      .replace('×', '*')
                      .replace('÷', '/')
                      .replace('−', '-')
                      .replace('–', '-')
                      .replace('—', '-')
                      .replace('^', '**')
                      .replace(',', '.')
                      )
        match = re.search(r'\d[\d\s+\-*/.()\[\]]*\d|\d', expression)
        ...
        if not match:
            return "Aucune expression mathématique trouvée."
        clean = match.group().strip()
        if not clean:
            return "Aucune expression mathématique trouvée."
        result = eval(clean, {"__builtins__": {}}, {})
        return f"{clean} = {result}"
    except ZeroDivisionError:
        return "Erreur : division par zéro impossible."
    except SyntaxError:
        return "Erreur : expression mathématique non reconnue."
    except Exception as e:
        return f"Erreur de calcul: {e}"