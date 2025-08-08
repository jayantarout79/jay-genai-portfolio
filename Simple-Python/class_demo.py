class character :
    def __init__(self, health,damage,speed):
        self.health = health
        self.damage = damage
        self.speed = speed  
    def double_speed(self):
        self.speed *= 2

warrior= character(100, 20, 10)
ninja= character(80, 15, 15)
print("Warrior's speed before doubling:", warrior.speed)
print("Ninja's speed before doubling:", ninja.speed)
warrior.double_speed()
print("Warrior's speed after doubling:", warrior.speed)
print("Ninja's speed after doubling:", ninja.speed)