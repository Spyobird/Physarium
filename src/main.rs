extern crate nannou;

use std::cmp;
use std::ops::Add;

use nannou::prelude::*;
use nannou::rand::random_range;

fn main() {
    nannou::app(model)
        .update(update)
        .run();
}

struct Model {
    universe: Universe
}

fn model(app: &App) -> Model {
    let _window = app.new_window().size(1024, 1024).view(view).build().unwrap();
    Model {
        universe: Universe::new(100000, 10, 1024, 1024, 5.0, 1.0, 12.0, 20.0, 34.0, 0.95, 10000.0)
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    model.universe.sense_step();
    model.universe.deposit();
    model.universe.diffuse();
}

fn view(app: &App, model: &Model, frame: Frame) {
    let _win = app.window_rect();
    /*app.set_loop_mode(LoopMode::NTimes {
        number_of_updates: 2
    });*/

    let _line_weight = 1.0;

    let draw = app.draw();

    if app.elapsed_frames() == 1 {
        draw.background().color(BLACK);
    }

    draw.rect()
        .w_h(1024.0, 1024.0)
        .color(rgba(0.0, 0.0, 0.0, 0.04));
    
    for agent in model.universe.agents.iter() {
        draw.rect()
            //.color(rgba(1.0, 1.0, 1.0, 0.1))
            .color(hsla((agent.angle / 360.0) as f32, 1.0, 0.27, 0.2))
            .x_y((agent.position.x as f32 - model.universe.width as f32 / 2.0 + 0.5) * 1.0, (agent.position.y as f32 - model.universe.height as f32 / 2.0 + 0.5) * 1.0)
            .w_h(1.5, 1.5);
    }

    /*for i in 0..model.universe.width {
        for j in 0..model.universe.height {
            let alpha = Universe::get_alpha(model.universe.map[Universe::xy_to_index(model.universe.width, model.universe.height, i, j)], model.universe.threshold);
            if alpha > 0.0 {
                draw.rect()
                    .color(rgba(1.0, 1.0, 1.0, alpha))
                    .x_y((i as f32 - model.universe.width as f32 / 2.0 + 0.5) * 1.0, (j as f32 - model.universe.height as f32 / 2.0 + 0.5) * 1.0)
                    .w_h(1.0, 1.0);
            }
        }
    }*/

    draw.to_frame(app, &frame).unwrap();

    let file_path = captured_frame_path(app, &frame);
    app.main_window().capture_frame(file_path);
}

struct Agent {
    position: DVec2,
    angle: f64,
    trail_length: usize,
    trail: Vec<DVec2>,
    trail_ptr: usize
}

impl Agent {
    fn new(x_pos: f64, y_pos: f64, angle: f64, trail_length: usize) -> Agent {
        Agent {
            position: DVec2::new(x_pos, y_pos),
            angle,
            trail_length,
            trail: Vec::with_capacity(trail_length),
            trail_ptr: 0
        }
    }

    fn step(&mut self, width: usize, height: usize, step_length: f64, step_angle: f64, angle_variance: f64, direction: i8) {
        /*if self.trail.len() < self.trail_length {
            self.trail.push(self.position.clone())
        } else {
            self.trail[self.trail_ptr] = self.position.clone();
        }
        self.trail_ptr += 1;
        self.trail_ptr %= self.trail_length;*/

        self.angle = match direction {
            -1 => Universe::angle_format(self.angle - (step_angle + random_range::<f64>(-angle_variance / 2.0, angle_variance / 2.0)) * random_range::<f64>(0.0, 1.0)),
            1 => Universe::angle_format(self.angle + (step_angle + random_range::<f64>(-angle_variance / 2.0, angle_variance / 2.0)) * random_range::<f64>(0.0, 1.0)),
            2 => Universe::angle_format(self.angle + (step_angle + random_range::<f64>(-angle_variance / 2.0, angle_variance / 2.0)) * random_range::<f64>(-1.0, 1.0)),
            _ => Universe::angle_format(self.angle + (random_range::<f64>(-angle_variance / 2.0, angle_variance / 2.0)) * random_range::<f64>(0.0, 1.0))
        };

        let mut add = Universe::vec_from_para(step_length, self.angle);
        let mut step = self.position.add(add);
        let mut within = Universe::within_bounds(width, height, &step);
        while !within {
            self.angle = random_range::<f64>(0.0, 360.0);
            add = Universe::vec_from_para(step_length, self.angle);
            step = self.position.add(add);
            within = Universe::within_bounds(width, height, &step);
        }
        self.position = step;

        //print!("{:?} {:?} {:?} {:?} {:?} {:?}", direction, add.to_string(), step.to_string(), self.angle, self.position.to_string(), within);
    }
}

struct Universe {
    agent_count: usize,
    agents: Vec<Agent>,
    width: usize,
    height: usize,
    map: Vec<f64>,
    diffuse_map: Vec<f64>,
    step_angle: f64,
    step_length: f64,
    angle_variance: f64,
    sensor_angle: f64,
    sensor_length: f64,
    attenuation_factor: f64,
    threshold: f64
}

impl Universe {
    fn new(agent_count: usize, trail_length: usize, width: usize, height: usize, step_angle: f64, step_length: f64, angle_variance: f64,
        sensor_angle: f64, sensor_length: f64, attenuation_factor: f64, threshold: f64) -> Self {
        let mut agents = Vec::with_capacity(agent_count);
        for _i in 0..agent_count {
            //let r = random_range::<f64>(0.0, cmp::min(width, height) as f64 / 3.0);
            let r = height as f64 / 3.0;
            let theta = random_range::<f64>(0.0, 360.0);
            agents.push(Agent::new(width as f64 / 2.0 + r * theta.to_radians().sin(), height as f64 / 2.0 + r * theta.to_radians().cos(), Universe::angle_format(theta - 180.0), trail_length))
            
            //agents.push(Agent::new(width as f64 / 2.0, height as f64 / 2.0, random_range::<f64>(0.0, 360.0), trail_length));
        }
        Universe {
            agent_count,
            agents,
            width,
            height,
            map: vec![0.0; width * height],
            diffuse_map: vec![0.0; width * height],
            step_angle,
            step_length,
            angle_variance,
            sensor_angle,
            sensor_length,
            attenuation_factor,
            threshold
        }
    }

    fn sense_step(&mut self) {
        for agent in self.agents.iter_mut() {
            let sense_left = agent.position.add(Universe::vec_from_para(self.sensor_length, Universe::angle_format(agent.angle - self.sensor_angle)));
            let left_val = if Universe::within_bounds(self.width, self.height, &sense_left) { self.map[Universe::coord_to_index(self.width, self.height, &sense_left)] } else { -1.0 };
            let sense_forward = agent.position.add(Universe::vec_from_para(self.sensor_length, Universe::angle_format(agent.angle)));
            let forward_val = if Universe::within_bounds(self.width, self.height, &sense_forward) { self.map[Universe::coord_to_index(self.width, self.height, &sense_forward)] } else { -1.0 };
            let sense_right = agent.position.add(Universe::vec_from_para(self.sensor_length, Universe::angle_format(agent.angle + self.sensor_angle)));
            let right_val = if Universe::within_bounds(self.width, self.height, &sense_right) { self.map[Universe::coord_to_index(self.width, self.height, &sense_right)] } else { -1.0 };
            agent.step(self.width, self.height, self.step_length, self.step_angle, self.angle_variance, Universe::cmp_direction(left_val, forward_val, right_val));
        }
    }

    fn deposit(&mut self) {
        for agent in self.agents.iter() {
            if Universe::within_bounds(self.width, self.height, &agent.position) {
                self.map[Universe::coord_to_index(self.width, self.height, &agent.position)] += 1.0;
            }
        }
    }

    fn diffuse(&mut self) {
        self.diffuse_map.fill(0.0);
        for i in 0..self.width * self.height {
            if self.map[i] > 0.00001 {
                let xy = Universe::index_to_xy(self.width, self.height, i);
                match Universe::check_neighbours(self.width, self.height, xy.0, xy.1) {
                    0 => {
                        let diffused_val = self.map[i] / 24.0;//4.0
                        for p in 0..4 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 4),
                                1 => Universe::get_neighbour(self.width, self.height, i, 5),
                                2 => Universe::get_neighbour(self.width, self.height, i, 7),
                                3 => Universe::get_neighbour(self.width, self.height, i, 8),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    1 => {
                        let diffused_val = self.map[i] / 18.0;//6.0
                        for p in 0..6 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 3),
                                1 => Universe::get_neighbour(self.width, self.height, i, 4),
                                2 => Universe::get_neighbour(self.width, self.height, i, 5),
                                3 => Universe::get_neighbour(self.width, self.height, i, 6),
                                4 => Universe::get_neighbour(self.width, self.height, i, 7),
                                5 => Universe::get_neighbour(self.width, self.height, i, 8),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    2 => {
                        let diffused_val = self.map[i] / 24.0;//4.0
                        for p in 0..4 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 3),
                                1 => Universe::get_neighbour(self.width, self.height, i, 4),
                                2 => Universe::get_neighbour(self.width, self.height, i, 6),
                                3 => Universe::get_neighbour(self.width, self.height, i, 7),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    3 => {
                        let diffused_val = self.map[i] / 28.0;//6.0
                        for p in 0..6 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 1),
                                1 => Universe::get_neighbour(self.width, self.height, i, 2),
                                2 => Universe::get_neighbour(self.width, self.height, i, 4),
                                3 => Universe::get_neighbour(self.width, self.height, i, 5),
                                4 => Universe::get_neighbour(self.width, self.height, i, 7),
                                5 => Universe::get_neighbour(self.width, self.height, i, 8),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    5 => {
                        let diffused_val = self.map[i] / 18.0;//6.0
                        for p in 0..6 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 0),
                                1 => Universe::get_neighbour(self.width, self.height, i, 1),
                                2 => Universe::get_neighbour(self.width, self.height, i, 3),
                                3 => Universe::get_neighbour(self.width, self.height, i, 4),
                                4 => Universe::get_neighbour(self.width, self.height, i, 6),
                                5 => Universe::get_neighbour(self.width, self.height, i, 7),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    6 => {
                        let diffused_val = self.map[i] / 24.0;//4.0
                        for p in 0..4 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 1),
                                1 => Universe::get_neighbour(self.width, self.height, i, 2),
                                2 => Universe::get_neighbour(self.width, self.height, i, 4),
                                3 => Universe::get_neighbour(self.width, self.height, i, 5),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    7 => {
                        let diffused_val = self.map[i] / 24.0;//4.0
                        for p in 0..6 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 0),
                                1 => Universe::get_neighbour(self.width, self.height, i, 1),
                                2 => Universe::get_neighbour(self.width, self.height, i, 2),
                                3 => Universe::get_neighbour(self.width, self.height, i, 3),
                                4 => Universe::get_neighbour(self.width, self.height, i, 4),
                                5 => Universe::get_neighbour(self.width, self.height, i, 5),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    8 => {
                        let diffused_val = self.map[i] / 24.0;//4.0
                        for p in 0..4 {
                            let j = match p {
                                0 => Universe::get_neighbour(self.width, self.height, i, 0),
                                1 => Universe::get_neighbour(self.width, self.height, i, 1),
                                2 => Universe::get_neighbour(self.width, self.height, i, 3),
                                3 => Universe::get_neighbour(self.width, self.height, i, 4),
                                _ => 0
                            };
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    },
                    _ => {
                        let diffused_val = self.map[i] / 9.0;
                        for p in 0..9 {
                            let j = Universe::get_neighbour(self.width, self.height, i, p);
                            self.diffuse_map[j] += diffused_val * self.attenuation_factor;
                        }
                    }
                };
            }
            // if self.map[i] <= threshold_value then set to 0
        }

        for i in 0..self.width * self.height {
            if self.diffuse_map[i] > self.threshold {
                self.diffuse_map[i] = self.threshold;
            }
        }
        self.map.copy_from_slice(&self.diffuse_map);
    }
}

impl Universe {
    fn coord_to_index(width: usize, height: usize, coord: &DVec2) -> usize {
        Universe::xy_to_index(width, height, coord.x.floor() as usize, coord.y.floor() as usize)
    }

    fn xy_to_index(width: usize, _height: usize, x: usize, y: usize) -> usize {
        y * width + x
    }

    fn index_to_xy(width: usize, _height: usize, index: usize) -> (usize, usize) {
        (index % width, index / width)
    }

    fn vec_from_para(d: f64, angle: f64) -> DVec2 {
        DVec2::new(d * angle.to_radians().sin(), d * angle.to_radians().cos())
    }

    fn angle_format(angle: f64) -> f64 {
        if angle < 0.0 {
            angle + 360.0
        } else if angle > 360.0 {
            angle - 360.0
        } else {
            angle
        }
    }

    fn within_bounds(width: usize, height: usize, coord: &DVec2) -> bool {
        if (coord.x < 0.0) || (coord.x > (width ) as f64) || (coord.y < 0.0) || (coord.y > (height ) as f64) {
            false
        } else {
            true
        }
    }

    fn cmp_direction(left: f64, forward: f64, right: f64) -> i8 {
        if (forward > left) & (forward > right) {
            0
        } else if (forward < left) & (forward < right) {
            2
        } else if left > right {
            -1
        } else if right > left {
            1
        } else {
            match random_range::<i8>(0, 3) {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0
            }
        }    
    }

    fn check_neighbours(width: usize, height: usize, x: usize, y: usize) -> u8 {
        if x == 0 {
            if y == 0 {
                0
            } else if y == height - 1 {
                6
            } else {
                3
            }
        } else if x == width - 1 {
            if y == 0 {
                2
            } else if y == height - 1 {
                8
            } else {
                5
            }
        } else {
            if y == 0 {
                1
            } else if y == height - 1 {
                7
            } else {
                4
            }
        }
    }

    fn get_neighbour(width: usize, _height: usize, index: usize, position: usize) -> usize {
        match position {
            0 => index - width - 1,
            1 => index - width,
            2 => index - width + 1,
            3 => index - 1,
            5 => index + 1,
            6 => index + width - 1,
            7 => index + width,
            8 => index + width + 1,
            _ => index
        }
    }

    fn get_alpha(value: f64, threshold: f64) -> f64 {
        let den = value / threshold;
        //let den = (52.0.pow(value / threshold) - 1.0) / 51.0;
        if value >= threshold {
            1.0
        } else if den <= 0.005 {
            0.0
        } else {
            den
        }
    }
}

fn captured_frame_path(app: &App, frame: &Frame) -> std::path::PathBuf {
    // Create a path that we want to save this frame to.
    app.project_path()
        .expect("failed to locate `project_path`")
        // Capture all frames to a directory called `/<path_to_nannou>/nannou/simple_capture`.
        .join(app.exe_name().unwrap())
        // Name each file after the number of the frame.
        .join(format!("{:05}", frame.nth()))
        // The extension will be PNG. We also support tiff, bmp, gif, jpeg, webp and some others.
        .with_extension("png")
}