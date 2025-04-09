#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
from gym import spaces
import pygame
import math
from collections import deque
import random as rd

class Vec2D:
    """Classe Vec2D similaire à sf::Vector2 des fichiers HPP."""
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        if isinstance(other, Vec2D):
            return Vec2D(self.x + other.x, self.y + other.y)
        return Vec2D(self.x + other, self.y + other)
    
    def __sub__(self, other):
        if isinstance(other, Vec2D):
            return Vec2D(self.x - other.x, self.y - other.y)
        return Vec2D(self.x - other, self.y - other)
    
    def __mul__(self, scalar):
        return Vec2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Vec2D(self.x / scalar, self.y / scalar)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def length_squared(self):
        return self.x**2 + self.y**2
    
    def normalize(self):
        length = self.length()
        if length > 0:
            self.x /= length
            self.y /= length
        return self
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def cross(self, other):
        return self.x * other.y - self.y * other.x
    
    def to_tuple(self):
        return (self.x, self.y)
    
    def to_numpy(self):
        return np.array([self.x, self.y])


class Object:
    """
    Implémentation de la classe Object du fichier object.hpp.
    Un Object représente un segment du pendule avec ses propriétés physiques.
    """
    def __init__(self):
        self.position = Vec2D(0.0, 0.0)
        self.position_last = Vec2D(0.0, 0.0)
        self.angle = 0.0
        self.angle_last = 0.0
        self.density = 1.0
        
        self.velocity = Vec2D(0.0, 0.0)
        self.angular_velocity = 0.0
        
        self.forces = Vec2D(0.0, 0.0)
        
        self.center_of_mass = Vec2D(0.0, 0.0)
        self.inv_mass = 1.0
        self.inv_inertia_tensor = 1.0
        
        self.particles = []  # Liste de Vec2D
    
    def compute_properties(self):
        """Calcule les propriétés de masse de l'objet."""
        pos_sum = Vec2D(0.0, 0.0)
        mass = 0.0
        
        for p in self.particles:
            pos_sum = pos_sum + p
            mass += 1.0
        
        if mass > 0:
            self.center_of_mass = pos_sum / mass
            self.inv_mass = 1.0 / (self.density * mass)
            
            # Calcul du tenseur d'inertie
            inertia_tensor = 0.0
            for p in self.particles:
                diff = p - self.center_of_mass
                inertia_tensor += (1.0 + diff.length_squared()) * self.density
            
            self.inv_inertia_tensor = 1.0 / inertia_tensor
    
    def update(self, dt):
        """Met à jour la position et l'angle de l'objet en fonction des forces."""
        # Mise à jour linéaire
        self.position_last = Vec2D(self.position.x, self.position.y)
        self.velocity = self.velocity + self.forces * (self.inv_mass * dt)
        self.position = self.position + self.velocity * dt
        
        # Mise à jour angulaire
        self.angle_last = self.angle
        self.angle = self.angle + self.angular_velocity * dt
    
    def update_velocities(self, dt, friction):
        """Met à jour les vitesses en fonction des nouvelles positions."""
        self.velocity = (self.position - self.position_last) / dt * (1.0 - friction)
        self.angular_velocity = (self.angle - self.angle_last) / dt * (1.0 - friction)
    
    def apply_position_correction(self, p, r):
        """Applique une correction de position."""
        self.position = self.position + p * self.inv_mass
        self.angle += r.cross(p) * self.inv_inertia_tensor
    
    def get_world_position(self, particle_idx=None):
        """Convertit une position locale en position mondiale."""
        if particle_idx is not None:
            if 0 <= particle_idx < len(self.particles):
                obj_coord = self.particles[particle_idx]
            else:
                return Vec2D(0.0, 0.0)
        else:
            obj_coord = self.center_of_mass
        
        # Transformation de coordonnées
        # 1. Translation vers la position de l'objet
        # 2. Rotation autour de l'angle de l'objet
        # 3. Translation pour centrer sur le centre de masse
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        
        # Coordonnées relatives au centre de masse
        rel_x = obj_coord.x - self.center_of_mass.x
        rel_y = obj_coord.y - self.center_of_mass.y
        
        # Rotation et translation
        rotated_x = rel_x * cos_angle - rel_y * sin_angle
        rotated_y = rel_x * sin_angle + rel_y * cos_angle
        
        world_x = rotated_x + self.position.x
        world_y = rotated_y + self.position.y
        
        return Vec2D(world_x, world_y)


class DragConstraint:
    """
    Implémentation simplifiée d'une contrainte de glissement.
    """
    def __init__(self):
        self.object = None    # Référence à un Object
        self.target = Vec2D(0.0, 0.0)  # Position cible
        self.compliance = 0.0001
        self.lambda_ = 0.0
    
    def create(self, obj, target, compliance=0.0001):
        """Configure la contrainte avec un objet et une cible."""
        self.object = obj
        self.target = Vec2D(target.x, target.y)
        self.compliance = compliance
    
    def solve(self, dt):
        """Résout la contrainte en appliquant une correction à l'objet."""
        if self.object is None:
            return
        
        # Calcule la différence entre la position actuelle et la cible
        obj_pos = self.object.get_world_position(0)
        diff = obj_pos - self.target
        
        # Force proportionnelle à la distance
        force = diff * (-10.0 / dt)
        
        # Applique la correction
        self.object.apply_position_correction(force * dt**2, Vec2D(0.0, 0.0))


class ObjectPinConstraint:
    """
    Implémentation simplifiée d'une contrainte d'ancrage entre deux objets.
    """
    class Anchor:
        def __init__(self, object_ref=None, particle_idx=0):
            self.object_ref = object_ref
            self.particle_idx = particle_idx
    
    def __init__(self):
        self.anchor1 = self.Anchor()
        self.anchor2 = self.Anchor()
        self.compliance = 0.0001
        self.lambda_ = 0.0
        self.rest_length = None
    
    def create(self, anchor1, anchor2, compliance=0.0001):
        """Configure la contrainte avec deux points d'ancrage."""
        self.anchor1 = anchor1
        self.anchor2 = anchor2
        self.compliance = compliance
        
        # Calcule la longueur au repos
        p1 = anchor1.object_ref.get_world_position(anchor1.particle_idx)
        p2 = anchor2.object_ref.get_world_position(anchor2.particle_idx)
        diff = p2 - p1
        self.rest_length = diff.length()
    
    def solve(self, dt):
        """Résout la contrainte en appliquant des corrections aux objets."""
        if self.anchor1.object_ref is None or self.anchor2.object_ref is None:
            return
        
        # Obtient les positions mondiales des points d'ancrage
        p1 = self.anchor1.object_ref.get_world_position(self.anchor1.particle_idx)
        p2 = self.anchor2.object_ref.get_world_position(self.anchor2.particle_idx)
        
        # Calcule la différence entre les positions actuelles
        diff = p2 - p1
        current_length = diff.length()
        
        if current_length < 0.0001:
            return  # Évite la division par zéro
        
        # Calcule la correction nécessaire pour maintenir la longueur au repos
        correction = (current_length - self.rest_length) / current_length
        
        # Direction de la correction
        direction = Vec2D(diff.x / current_length, diff.y / current_length)
        
        # Calcule les forces à appliquer à chaque objet
        force1 = direction * correction * 0.5
        force2 = direction * -correction * 0.5
        
        # Applique les corrections
        self.anchor1.object_ref.apply_position_correction(force1 * dt**2, Vec2D(0.0, 0.0))
        self.anchor2.object_ref.apply_position_correction(force2 * dt**2, Vec2D(0.0, 0.0))


class Solver:
    """
    Implémentation de la classe Solver du fichier solver.hpp.
    Le Solver intègre le mouvement des objets et résout les contraintes.
    """
    def __init__(self):
        self.objects = []  # Liste d'Object
        self.drag_constraints = []  # Liste de DragConstraint
        self.object_pins = []  # Liste d'ObjectPinConstraint
        
        self.gravity = Vec2D(0.0, 1000.0)
        self.friction = 0.0
        self.sub_steps = 2
    
    def update(self, dt):
        """Met à jour la simulation en solvant les contraintes de position."""
        pos_iter = 1
        sub_dt = dt / self.sub_steps
        
        for _ in range(self.sub_steps):
            # Mise à jour des forces et positions
            for obj in self.objects:
                obj.forces = self.gravity * (1.0 / obj.inv_mass)
                obj.update(sub_dt)
            
            # Réinitialiser les contraintes
            self.reset_constraints()
            
            # Résoudre les contraintes
            for _ in range(pos_iter):
                self.solve_constraints(sub_dt)
            
            # Mise à jour des vitesses
            for obj in self.objects:
                obj.update_velocities(sub_dt, self.friction)
    
    def create_object(self):
        """Crée un nouvel objet et le retourne."""
        obj = Object()
        self.objects.append(obj)
        return obj
    
    def create_drag_constraint(self, obj, target, compliance=0.0001):
        """Crée une contrainte de glissement."""
        constraint = DragConstraint()
        constraint.create(obj, target, compliance)
        self.drag_constraints.append(constraint)
        return constraint
    
    def create_object_pin_constraint(self, anchor1, anchor2, compliance=0.0001):
        """Crée une contrainte d'ancrage entre deux objets."""
        constraint = ObjectPinConstraint()
        constraint.create(anchor1, anchor2, compliance)
        self.object_pins.append(constraint)
        return constraint
    
    def reset_constraints(self):
        """Réinitialise les multiplicateurs de Lagrange des contraintes."""
        for c in self.drag_constraints:
            c.lambda_ = 0.0
        
        for c in self.object_pins:
            c.lambda_ = 0.0
    
    def solve_constraints(self, dt):
        """Résout toutes les contraintes."""
        for c in self.drag_constraints:
            c.solve(dt)
        
        for c in self.object_pins:
            c.solve(dt)


class TriplePendulumEnvPBD(gym.Env):
    """
    Environnement Gym pour un triple pendule sur un chariot,
    implémenté avec une physique Position-Based Dynamics
    similaire à celle des fichiers HPP.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode=None, num_nodes=3):
        super(TriplePendulumEnvPBD, self).__init__()
        
        self.num_nodes = num_nodes  # Nombre de segments du pendule
        
        # Paramètres de l'environnement
        self.segment_size = 100.0  # Taille d'un segment du pendule
        self.slider_length = 500.0  # Longueur du rail du chariot
        self.world_size = Vec2D(
            self.slider_length + 2.2 * self.num_nodes * self.segment_size,
            self.num_nodes * self.segment_size * 2.25
        )
        
        # Paramètres de simulation
        self.solver = Solver()
        self.solver.gravity = Vec2D(0.0, 1000.0)
        self.solver.friction = 0.05
        self.solver.sub_steps = 4
        
        # Paramètres de contrôle
        self.force_mag = 500.0  # Force maximale applicable
        self.max_speed = 300.0  # Vitesse maximale du chariot
        self.max_accel = 500.0  # Accélération maximale
        
        # État
        self.agent = None  # L'agent sera initialisé dans reset()
        self.current_velocity = 0.0
        self.current_time = 0.0
        self.freeze_time = 1.0  # Temps pendant lequel le pendule est gelé au début
        
        # Compliance (inverse de rigidité) des contraintes
        self.compliance = 0.0001
        
        # Limites
        self.x_threshold = self.slider_length / 2
        self.x_dot_threshold = self.max_speed * 1.5
        
        # Espace d'action: force appliquée au chariot
        self.action_space = spaces.Box(
            low=-self.force_mag, 
            high=self.force_mag, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Espace d'observation: état complet du système
        # [x, x_dot, angles, angular_velocities]
        obs_dimension = 2 + 2 * self.num_nodes
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dimension,), 
            dtype=np.float32
        )
        
        # Rendu
        self.render_mode = render_mode
        self.screen = None
        self.screen_width = 800
        self.screen_height = 600
        self.cart_y_pos = self.screen_height // 2
        self.pixels_per_meter = 2.0
        self.clock = None
        self.font = None
        
        # Performance tracking
        self.consecutive_upright_steps = 0
    
    def initialize_agent(self):
        """Initialise l'agent avec le moteur physique PBD."""
        # Réinitialisation du solver
        self.solver = Solver()
        self.solver.gravity = Vec2D(0.0, 1000.0)
        self.solver.friction = 0.05
        self.solver.sub_steps = 4
        
        # Création des segments du pendule
        last_segment = None
        for i in range(self.num_nodes):
            segment = self.solver.create_object()
            
            # Ajoute deux particules à chaque segment: début et fin
            segment.particles.append(Vec2D(0.0, 0.0))
            segment.particles.append(Vec2D(self.segment_size, 0.0))
            
            # Calcule les propriétés de masse
            segment.compute_properties()
            
            # Positionne le segment
            segment.position = Vec2D(
                self.world_size.x * 0.5,
                (0.5 + float(i)) * self.segment_size
            )
            segment.angle = math.pi * 0.5  # Angle initial (droit vers le haut)
            
            # Si ce n'est pas le premier segment, le relier au précédent
            if last_segment:
                self.solver.create_object_pin_constraint(
                    ObjectPinConstraint.Anchor(last_segment, 1),
                    ObjectPinConstraint.Anchor(segment, 0),
                    self.compliance
                )
            else:
                # Premier segment: contrainte de glissement sur le chariot
                drag = self.solver.create_drag_constraint(
                    segment, 
                    segment.get_world_position(0),
                    self.compliance
                )
                self.cart_constraint = drag  # Sauvegarde pour pouvoir la manipuler plus tard
            
            last_segment = segment
        
        return True
    
    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement à un état initial."""
        if seed is not None:
            np.random.seed(seed)
        
        # Réinitialise les variables de temps
        self.current_time = 0.0
        self.current_velocity = 0.0
        
        # Initialise l'agent
        self.initialize_agent()
        
        # Ajoute une perturbation aléatoire initiale
        rd_angle = rd.uniform(-0.1, 0.1)
        for i, obj in enumerate(self.solver.objects):
            obj.angle = math.pi * 0.5 + rd_angle * (i + 1)
        
        # Initialise le rendu si nécessaire
        if self.render_mode == "human" and self.screen is None:
            self._render_init()
        
        # Renvoie l'observation initiale
        return self.get_observation(), {}
    
    def get_observation(self):
        """Récupère l'observation actuelle du système."""
        if not self.solver.objects:
            return np.zeros(2 + 2 * self.num_nodes, dtype=np.float32)
        
        # Position et vitesse du chariot
        cart_x = self.cart_constraint.target.x - self.world_size.x * 0.5
        cart_x_dot = self.current_velocity
        
        # Angles et vitesses angulaires des segments
        angles = []
        angular_velocities = []
        
        for obj in self.solver.objects:
            # Normaliser l'angle entre -π et π
            angle = obj.angle % (2 * math.pi)
            if angle > math.pi:
                angle -= 2 * math.pi
            
            angles.append(angle)
            angular_velocities.append(obj.angular_velocity)
        
        # Combiner toutes les observations
        observation = np.array(
            [cart_x, cart_x_dot] + angles + angular_velocities,
            dtype=np.float32
        )
        
        return observation
    
    def step(self, action):
        """Exécute une étape de simulation avec l'action donnée."""
        # Extraire et limiter la force
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        
        # Temps de simulation pour cette étape
        dt = 0.02  # 50 Hz
        
        # Si dans la période de gel, sauter la mise à jour
        if self.current_time < self.freeze_time:
            self.current_time += dt
            return self.get_observation(), 0.0, False, False, {}
        
        # Mettre à jour la vitesse en fonction de l'action
        self.current_velocity += force * dt
        
        # Limiter la vitesse
        if self.current_velocity > self.max_speed:
            self.current_velocity = self.max_speed
        elif self.current_velocity < -self.max_speed:
            self.current_velocity = -self.max_speed
        
        # Mettre à jour la position du chariot
        cart_pos = self.cart_constraint.target
        cart_pos.x += self.current_velocity * dt
        
        # Limiter la position du chariot
        min_pos = self.world_size.x * 0.5 - self.slider_length * 0.5
        max_pos = self.world_size.x * 0.5 + self.slider_length * 0.5
        
        if cart_pos.x < min_pos:
            cart_pos.x = min_pos
            self.current_velocity = 0.0
        if cart_pos.x > max_pos:
            cart_pos.x = max_pos
            self.current_velocity = 0.0
        
        # Mettre à jour la position cible de la contrainte du chariot
        self.cart_constraint.target = cart_pos
        
        # Mettre à jour la simulation physique
        self.solver.update(dt)
        
        # Mettre à jour le temps
        self.current_time += dt
        
        # Vérifier les conditions de terminaison
        terminated = False
        
        # Vérifier si le pendule est debout
        is_upright = self.is_pendulum_upright()
        
        if is_upright:
            self.consecutive_upright_steps += 1
        else:
            self.consecutive_upright_steps = 0
        
        # Récompense
        reward = self.calculate_reward(is_upright)
        
        # Info supplémentaire
        info = {
            "is_upright": is_upright,
            "consecutive_upright_steps": self.consecutive_upright_steps
        }
        
        return self.get_observation(), reward, terminated, False, info
    
    def is_pendulum_upright(self):
        """Vérifie si le pendule est en position verticale."""
        if not self.solver.objects:
            return False
        
        # Un pendule est considéré debout si tous ses segments sont proches de la verticale
        for obj in self.solver.objects:
            # Vérifier l'angle par rapport à la verticale (π/2)
            angle_diff = abs((obj.angle % (2 * math.pi)) - math.pi * 0.5)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            if angle_diff > 0.3:  # ~17 degrés de tolérance
                return False
        
        return True
    
    def calculate_reward(self, is_upright):
        """Calcule la récompense en fonction de l'état actuel."""
        if not self.solver.objects:
            return 0.0
        
        reward = 0.0
        
        # Pénalité pour l'écart par rapport à la verticale
        for obj in self.solver.objects:
            # Calculer l'écart par rapport à la verticale (π/2)
            angle_diff = abs((obj.angle % (2 * math.pi)) - math.pi * 0.5)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            # Récompense inversement proportionnelle à l'écart
            reward -= angle_diff * 0.1
        
        # Bonus pour être debout
        if is_upright:
            reward += 1.0
        
        # Bonus pour maintenir la position debout
        if self.consecutive_upright_steps > 0:
            reward += min(self.consecutive_upright_steps / 100.0, 1.0)
        
        # Pénalité pour être loin du centre
        cart_x = self.cart_constraint.target.x - self.world_size.x * 0.5
        reward -= abs(cart_x) * 0.001
        
        return reward
    
    def render(self):
        """Rend l'environnement graphiquement."""
        if self.render_mode != "human":
            return
        
        if self.screen is None:
            self._render_init()
        
        # Couleurs
        BACKGROUND_COLOR = (240, 240, 245)
        CART_COLOR = (50, 50, 60)
        TRACK_COLOR = (180, 180, 190)
        PENDULUM_COLORS = [
            (220, 60, 60),   # Rouge pour le premier segment
            (60, 180, 60),   # Vert pour le deuxième
            (60, 60, 220)    # Bleu pour le troisième
        ]
        TEXT_COLOR = (40, 40, 40)
        
        # Nettoyer l'écran
        self.screen.fill(BACKGROUND_COLOR)
        
        # Dessiner le rail
        track_height = 5
        track_width = self.slider_length * self.pixels_per_meter
        pygame.draw.rect(
            self.screen,
            TRACK_COLOR,
            pygame.Rect(
                self.screen_width // 2 - track_width // 2,
                self.cart_y_pos + 15,
                track_width,
                track_height
            )
        )
        
        # Dessiner le chariot
        cart_x = self.cart_constraint.target.x - self.world_size.x * 0.5
        cart_x_px = int(self.screen_width / 2 + cart_x * self.pixels_per_meter)
        cart_y_px = self.cart_y_pos
        
        cart_w, cart_h = 60, 30
        cart_rect = pygame.Rect(
            cart_x_px - cart_w//2,
            cart_y_px - cart_h//2,
            cart_w,
            cart_h
        )
        pygame.draw.rect(self.screen, CART_COLOR, cart_rect, border_radius=5)
        
        # Dessiner les segments du pendule
        for i, obj in enumerate(self.solver.objects):
            if i >= len(PENDULUM_COLORS):
                color = (100, 100, 100)  # Gris pour segments supplémentaires
            else:
                color = PENDULUM_COLORS[i]
            
            # Obtenir les positions dans l'espace de l'écran
            p0 = obj.get_world_position(0)
            p1 = obj.get_world_position(1)
            
            # Convertir en pixels
            p0x = int(self.screen_width / 2 + (p0.x - self.world_size.x * 0.5) * self.pixels_per_meter)
            p0y = int(self.cart_y_pos + (p0.y - self.world_size.y * 0.5) * self.pixels_per_meter)
            p1x = int(self.screen_width / 2 + (p1.x - self.world_size.x * 0.5) * self.pixels_per_meter)
            p1y = int(self.cart_y_pos + (p1.y - self.world_size.y * 0.5) * self.pixels_per_meter)
            
            # Dessiner le segment
            pygame.draw.line(self.screen, color, (p0x, p0y), (p1x, p1y), 6)
            
            # Dessiner les articulations
            pygame.draw.circle(self.screen, (30, 30, 30), (p0x, p0y), 6)
            pygame.draw.circle(self.screen, (30, 30, 30), (p1x, p1y), 6)
        
        # Afficher des informations
        if self.font is None:
            self.font = pygame.font.Font(None, 24)
        
        # Texte d'information
        text_lines = [
            f"Position: {cart_x:.2f}",
            f"Vitesse: {self.current_velocity:.2f}",
            f"Temps: {self.current_time:.2f}",
            f"Upright Steps: {self.consecutive_upright_steps}"
        ]
        
        for i, line in enumerate(text_lines):
            text = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (10, 10 + i * 25))
        
        # Mise à jour de l'affichage
        pygame.display.flip()
        
        # Limiter FPS
        if self.clock:
            self.clock.tick(60)
    
    def close(self):
        """Ferme la fenêtre de rendu."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
    
    def _render_init(self):
        """Initialise le rendu Pygame."""
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("Triple Pendulum PBD")
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Arial", 16)
        except:
            self.font = pygame.font.Font(None, 16)


# Test simple
if __name__ == "__main__":
    env = TriplePendulumEnvPBD(render_mode="human")
    obs = env.reset()[0]
    
    done = False
    total_reward = 0
    
    while not done:
        # Action aléatoire
        action = env.action_space.sample()
        
        # Exécuter une étape
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        env.render()
        
        # Pause pour ralentir la simulation
        pygame.time.wait(10)
    
    print(f"Total reward: {total_reward}")
    env.close()
