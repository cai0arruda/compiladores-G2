def comerWaffle(poder):
    return poder + 1
def main():
    poder = 0
    lifeVecna = 10
    nome = input("Escolha seu personagem (Dustin - Mike - Lukas - Will): ")
    while poder < lifeVecna:
        print(f"Você precisa de mais poder {nome}, coma waffles - Seu poder Atual: {poder}")
        poder = comerWaffle(poder)
        if poder >= lifeVecna:
            print(f"{nome}, você derrotou o Vecna!")
        else:
            print("Precisa comer mais waffles para derrotar o Vecna!")
if __name__ == "__main__":
    main()
