class Sheep:
    def __init__(self, energy, Denergy):
        self.energy = energy
        self.Denergy = Denergy

    def make_sound(self):
        print("Baa")

class SheepWithGender(Sheep):
    def __init__(self, energy, Denergy, sex):
        super().__init__(energy, Denergy)
        self.sex = sex
