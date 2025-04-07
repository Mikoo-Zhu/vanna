from vpython import *
import numpy as np
import random

# --- 配置 ---
scene.width = 800
scene.height = 600
scene.title = "3D 碰撞模拟 (20个球体)"
scene.camera.pos = vector(0, 0, 15) # 第三人称视角
scene.camera.axis = vector(0, 0, -15)

# --- 常量 ---
container_size = 5.0  # 容器半边长
wall_thickness = 0.1
wall_transparency = 0.1
num_balls = 20
ball_radius = 0.3
max_initial_speed = 1.0
dt = 0.01 # 时间步长
coeff_restitution = 0.95 # 碰撞恢复系数 (近似弹性碰撞)

# --- 创建容器 ---
L = container_size
wall_color = color.gray(0.7)

wall_right = box(pos=vector(L, 0, 0), size=vector(wall_thickness, 2*L, 2*L), color=wall_color, opacity=wall_transparency)
wall_left = box(pos=vector(-L, 0, 0), size=vector(wall_thickness, 2*L, 2*L), color=wall_color, opacity=wall_transparency)
wall_top = box(pos=vector(0, L, 0), size=vector(2*L, wall_thickness, 2*L), color=wall_color, opacity=wall_transparency)
wall_bottom = box(pos=vector(0, -L, 0), size=vector(2*L, wall_thickness, 2*L), color=wall_color, opacity=wall_transparency)
wall_back = box(pos=vector(0, 0, -L), size=vector(2*L, 2*L, wall_thickness), color=wall_color, opacity=wall_transparency)
# 前墙透明以便观察
# wall_front = box(pos=vector(0, 0, L), size=vector(2*L, 2*L, wall_thickness), color=wall_color, opacity=1)

# --- 创建球体 ---
balls = []
for i in range(num_balls):
    # 确保初始位置不重叠且在容器内
    while True:
        pos = vector(
            random.uniform(-L + ball_radius, L - ball_radius),
            random.uniform(-L + ball_radius, L - ball_radius),
            random.uniform(-L + ball_radius, L - ball_radius)
        )
        valid_pos = True
        for existing_ball in balls:
            if mag(pos - existing_ball.pos) < 2 * ball_radius:
                valid_pos = False
                break
        if valid_pos:
            break

    ball = sphere(
        pos=pos,
        radius=ball_radius,
        color=vector(random.random(), random.random(), random.random()),
        make_trail=False, # 可以设为 True 来观察轨迹
        trail_radius=0.05
    )
    ball.velocity = vector(
        random.uniform(-max_initial_speed, max_initial_speed),
        random.uniform(-max_initial_speed, max_initial_speed),
        random.uniform(-max_initial_speed, max_initial_speed)
    )
    ball.mass = 1.0 # 假设所有球质量相同
    balls.append(ball)

# --- 碰撞检测与处理函数 ---
def handle_wall_collisions(ball):
    """处理球与墙壁的碰撞"""
    if abs(ball.pos.x) + ball.radius > L:
        ball.velocity.x = -ball.velocity.x * coeff_restitution
        # 防止卡墙
        ball.pos.x = L * sign(ball.pos.x) - ball.radius * sign(ball.pos.x)
    if abs(ball.pos.y) + ball.radius > L:
        ball.velocity.y = -ball.velocity.y * coeff_restitution
        ball.pos.y = L * sign(ball.pos.y) - ball.radius * sign(ball.pos.y)
    if abs(ball.pos.z) + ball.radius > L:
        ball.velocity.z = -ball.velocity.z * coeff_restitution
        ball.pos.z = L * sign(ball.pos.z) - ball.radius * sign(ball.pos.z)

def handle_ball_collisions(balls_list):
    """处理球与球之间的碰撞"""
    for i in range(len(balls_list)):
        for j in range(i + 1, len(balls_list)):
            ball1 = balls_list[i]
            ball2 = balls_list[j]
            dist_vec = ball1.pos - ball2.pos
            dist_mag = mag(dist_vec)

            if dist_mag < (ball1.radius + ball2.radius):
                # 检测到碰撞
                # 法线方向 (从 ball2 指向 ball1)
                normal_vec = norm(dist_vec)
                # 相对速度
                relative_vel = ball1.velocity - ball2.velocity
                # 沿法线方向的相对速度分量
                vel_along_normal = dot(relative_vel, normal_vec)

                # 如果球体正在相互靠近才处理碰撞
                if vel_along_normal < 0:
                    # 计算冲量 (基于恢复系数)
                    impulse_scalar = -(1 + coeff_restitution) * vel_along_normal
                    impulse_scalar /= (1/ball1.mass + 1/ball2.mass) # 考虑质量

                    # 计算冲量矢量
                    impulse_vec = impulse_scalar * normal_vec

                    # 更新速度
                    ball1.velocity += impulse_vec / ball1.mass
                    ball2.velocity -= impulse_vec / ball2.mass

                    # --- 分离重叠 ---
                    overlap = (ball1.radius + ball2.radius) - dist_mag
                    separation_vec = overlap * normal_vec * 0.5 # 各自分离一半
                    ball1.pos += separation_vec
                    ball2.pos -= separation_vec


# --- 模拟主循环 ---
while True:
    rate(100) # 控制模拟速度 (帧率)

    # 更新所有球的位置
    for ball in balls:
        ball.pos += ball.velocity * dt

    # 处理碰撞
    for ball in balls:
        handle_wall_collisions(ball)
    handle_ball_collisions(balls)
