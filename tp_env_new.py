import numpy as np
import math

# Définition de RealType
RealType = float

# Fonctions utilitaires

def cross2d(a, b):
    """Produit vectoriel 2D de deux vecteurs a et b."""
    return a[0]*b[1] - a[1]*b[0]

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def get_generalized_inv_mass(obj, r, n):
    """Calcule la masse généralisée inverse: inv_mass + (cross(r, n))^2 * inv_inertia_tensor."""
    return obj.inv_mass + (cross2d(r, n) ** 2) * obj.inv_inertia_tensor

# Structure de contrainte
class Constraint:
    def __init__(self, compliance=0.0001):
        self.lambda_val = 0.0
        self.compliance = compliance
        self.force = 0.0

# Structure d'ancrage
class Anchor:
    def __init__(self, obj=None, obj_coord=None):
        self.obj = obj
        self.obj_coord = obj_coord if obj_coord is not None else np.array([0.0, 0.0])

# Classe Object reprenant les principes de physics_guide/Pendulum-NEAT-main/src/user/common/physic/object.hpp
class Object:
    def __init__(self):
        self.position = np.array([0.0, 0.0])
        self.position_last = np.array([0.0, 0.0])
        self.angle = 0.0
        self.angle_last = 0.0
        self.density = 1.0
        self.velocity = np.array([0.0, 0.0])
        self.angular_velocity = 0.0
        self.forces = np.array([0.0, 0.0])
        self.center_of_mass = np.array([0.0, 0.0])
        self.inv_mass = 1.0
        self.inv_inertia_tensor = 1.0
        self.particles = []  # Liste de vecteurs (np.array) définissant la forme de l'objet
        self.color = (255, 255, 255)  # Couleur, non utilisé en physique

    def compute_properties(self):
        """Calcule le centre de masse, l'inverse de la masse et du tenseur d'inertie."""
        if not self.particles:
            return
        sum_pos = np.zeros(2)
        for p in self.particles:
            sum_pos += p
        self.center_of_mass = sum_pos / len(self.particles)
        mass = len(self.particles)
        self.inv_mass = 1.0 / (self.density * mass)
        inertia_tensor = 0.0
        for p in self.particles:
            inertia_tensor += (1.0 + np.linalg.norm(p - self.center_of_mass)**2) * self.density
        self.inv_inertia_tensor = 1.0 / inertia_tensor

    def update(self, dt):
        """Mise à jour linéaire et angulaire de la position de l'objet."""
        self.position_last = self.position.copy()
        self.velocity = self.velocity + dt * (self.forces * self.inv_mass)
        self.position = self.position + self.velocity * dt
        self.angle_last = self.angle
        self.angle = self.angle + self.angular_velocity * dt

    def update_velocities(self, dt, friction):
        """Mise à jour des vitesses linéaires et angulaires en appliquant la friction."""
        self.velocity = ((self.position - self.position_last) / dt) * (1.0 - friction)
        self.angular_velocity = ((self.angle - self.angle_last) / dt) * (1.0 - friction)

    def apply_position_correction(self, p, r):
        """Applique la correction de position et met à jour la rotation de l'objet."""
        self.position += p * self.inv_mass
        self.angle += cross2d(r, p) * self.inv_inertia_tensor

    def apply_rotation_correction(self, a):
        """Applique une correction de rotation."""
        self.angle += a * self.inv_inertia_tensor

    def get_world_position(self, obj_coord):
        """Calcule la position dans l'espace monde à partir d'une coordonnée de l'objet."""
        rel = obj_coord - self.center_of_mass
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        rotated = np.array([cos_a * rel[0] - sin_a * rel[1],
                            sin_a * rel[0] + cos_a * rel[1]])
        return self.position + rotated

    def get_object_position(self, world_coord):
        """Transforme une coordonnée monde en coordonnée objet."""
        rel = world_coord - self.position
        cos_a = math.cos(-self.angle)
        sin_a = math.sin(-self.angle)
        rotated = np.array([cos_a * rel[0] - sin_a * rel[1],
                            sin_a * rel[0] + cos_a * rel[1]])
        return rotated + self.center_of_mass

# Classe DragConstraint reprenant physics_guide/Pendulum-NEAT-main/src/user/common/physic/constraints/drag_constraint.hpp
class DragConstraint:
    def __init__(self, compliance=0.0001):
        self.constraint = Constraint(compliance)
        self.target = np.array([0.0, 0.0])
        self.anchor = Anchor()

    def create(self, obj, a):
        # Initialise l'ancrage en utilisant la position objet calculée
        self.anchor.obj = obj
        self.anchor.obj_coord = obj.get_object_position(a)

    def solve(self, dt):
        pa = self.anchor.obj.get_world_position(self.anchor.obj_coord)
        r1 = pa - self.anchor.obj.position
        v = self.target - pa
        d = np.linalg.norm(v)
        if d == 0:
            return
        n = v / d
        w1 = get_generalized_inv_mass(self.anchor.obj, r1, n)
        w2 = 0.0
        a_compliance = self.constraint.compliance / (dt * dt)
        delta_lambda = d / (w1 + w2 + a_compliance)
        p = delta_lambda * n
        self.anchor.obj.apply_position_correction(p, r1)
        self.constraint.force = self.constraint.lambda_val / (dt * dt) if dt != 0 else 0

# Classe ObjectPinConstraint reprenant physics_guide/Pendulum-NEAT-main/src/user/common/physic/constraints/object_pin.hpp
class ObjectPinConstraint:
    def __init__(self, compliance=0.0001):
        self.constraint = Constraint(compliance)
        self.anchor_1 = Anchor()
        self.anchor_2 = Anchor()

    def create(self, anchor_1, anchor_2):
        self.anchor_1 = anchor_1
        self.anchor_2 = anchor_2

    def solve(self, dt):
        a1_world = self.anchor_1.obj.get_world_position(self.anchor_1.obj_coord)
        a2_world = self.anchor_2.obj.get_world_position(self.anchor_2.obj_coord)
        r1 = a1_world - self.anchor_1.obj.position
        r2 = a2_world - self.anchor_2.obj.position
        v = a1_world - a2_world
        d = np.linalg.norm(v)
        if d == 0:
            return
        n = v / d
        w1 = get_generalized_inv_mass(self.anchor_1.obj, r1, n)
        w2 = get_generalized_inv_mass(self.anchor_2.obj, r2, -n)
        a_compliance = self.constraint.compliance / (dt * dt)
        delta_lambda = (d - a_compliance * self.constraint.lambda_val) / (w1 + w2 + a_compliance)
        self.constraint.lambda_val += delta_lambda
        p = delta_lambda * n
        self.anchor_1.obj.apply_position_correction(-p, r1)
        self.anchor_2.obj.apply_position_correction(p, r2)
        self.constraint.force = self.constraint.lambda_val / (dt * dt) if dt != 0 else 0

# Classe Solver reprenant physics_guide/Pendulum-NEAT-main/src/user/common/physic/solver.hpp
class Solver:
    def __init__(self, sub_steps=2, gravity=np.array([0.0, 1000.0]), friction=0.0):
        self.objects = []
        self.drag_constraints = []
        self.object_pins = []
        self.gravity = gravity
        self.friction = friction
        self.sub_steps = sub_steps

    def update(self, dt):
        pos_iter = 1
        sub_dt = dt / self.sub_steps
        for _ in range(self.sub_steps):
            # Mise à jour des objets avec la force de gravité
            for obj in self.objects:
                # Equivalence de: obj.forces = gravity / obj.inv_mass
                obj.forces = self.gravity / obj.inv_mass
                obj.update(sub_dt)
            self.reset_constraints()
            for _ in range(pos_iter):
                self.solve_constraints(sub_dt)
            for obj in self.objects:
                obj.update_velocities(sub_dt, self.friction)

    def create_drag_constraint(self, obj, target, compliance=0.0001):
        dc = DragConstraint(compliance)
        dc.create(obj, target)
        dc.target = target
        self.drag_constraints.append(dc)
        return dc

    def remove_drag_constraint(self, dc):
        if dc in self.drag_constraints:
            self.drag_constraints.remove(dc)

    def create_object_pin_constraint(self, anchor_1, anchor_2, compliance=0.0001):
        opc = ObjectPinConstraint(compliance)
        opc.create(anchor_1, anchor_2)
        self.object_pins.append(opc)
        return opc

    def remove_object_pin_constraint(self, opc):
        if opc in self.object_pins:
            self.object_pins.remove(opc)

    def create_object(self):
        obj = Object()
        self.objects.append(obj)
        return obj

    def reset_constraints(self):
        for dc in self.drag_constraints:
            dc.constraint.lambda_val = 0.0
        for opc in self.object_pins:
            opc.constraint.lambda_val = 0.0

    def solve_constraints(self, dt):
        for dc in self.drag_constraints:
            dc.solve(dt)
        for opc in self.object_pins:
            opc.solve(dt)

# Exemple d'utilisation si lancé directement
if __name__ == "__main__":
    # Création d'un Solver avec 4 sous-étapes, gravité modifiée et faible friction
    solver = Solver(sub_steps=4, gravity=np.array([0.0, 980.0]), friction=0.01)
    
    # Création d'un objet physique représentant potentiellement un pendule
    obj = solver.create_object()
    obj.position = np.array([100.0, 100.0])
    # Définir quelques particules pour définir la forme de l'objet
    obj.particles = [np.array([100.0, 100.0]), np.array([110.0, 100.0]), np.array([105.0, 110.0])]
    obj.compute_properties()
    
    # Création d'une contrainte de traînée
    target = np.array([120.0, 120.0])
    solver.create_drag_constraint(obj, target, compliance=0.001)
    
    # Simulation sur environ 1 seconde (60 itérations)
    dt = 0.016  # Intervalle de temps (en secondes)
    for i in range(60):
        solver.update(dt)
        print(f"Étape {i}: Position = {obj.position}, Angle = {obj.angle}")
