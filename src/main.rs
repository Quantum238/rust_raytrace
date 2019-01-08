use std::fs::File;
use std::io::prelude::*;
use std::ops::{Add, Sub, Index, AddAssign, SubAssign, Mul, Div, MulAssign, DivAssign};
// use std::path::Path;

#[derive(Debug, Copy, Clone)]
struct Vec3 {
	x: f64,
	y: f64,
	z: f64,
}

impl AddAssign for Vec3 {
	fn add_assign(&mut self, rhs: Vec3){
		self.x += rhs.x;
		self.y += rhs.y;
		self.z += rhs.z;
	}
}
impl SubAssign for Vec3 {
	fn sub_assign(&mut self, rhs: Vec3){
		self.x -= rhs.x;
		self.y -= rhs.y;
		self.z -= rhs.z;
	}
}
impl MulAssign<Vec3> for Vec3 {
	fn mul_assign(&mut self, rhs: Vec3){
		self.x *= rhs.x;
		self.y *= rhs.y;
		self.z *= rhs.z;
	}
}
impl MulAssign<f64> for Vec3 {
	fn mul_assign(&mut self, rhs: f64){
		self.x *= rhs;
		self.y *= rhs;
		self.z *= rhs;
	}
}
impl DivAssign<Vec3> for Vec3 {
	fn div_assign(&mut self, rhs: Vec3){
		self.x /= rhs.x;
		self.y /= rhs.y;
		self.z /= rhs.z;
	}
}
impl DivAssign<f64> for Vec3 {
	fn div_assign(&mut self, t: f64){
		let k = 1.0 / t;
		self.x *= k;
		self.y *= k;
		self.z *= k;
	}
}

impl Index<i32> for Vec3{
	type Output = f64;

	fn index(&self, idx: i32) -> &f64{
		match idx {
			0 => &self.x,
			1 => &self.y,
			2 => &self.z,
			_ => panic!("Index better, idiot"),
		}
	}
}
impl Div<Vec3> for Vec3 {
	type Output = Vec3;
	fn div(self, rhs: Self) -> Self {
		Vec3 {
			x: self.x / rhs.x,
			y: self.y / rhs.y,
			z: self.z / rhs.z,
		}
	}
}

impl Div<f64> for Vec3 {
	type Output = Vec3;
	fn div(self, t: f64) -> Self {
		Vec3 {
			x: self.x / t,
			y: self.y / t,
			z: self.z / t,
		}
	}
}
impl Mul<Vec3> for Vec3 {
	type Output = Vec3;
	fn mul(self, rhs: Self) -> Self {
		Vec3 {
			x: self.x * rhs.x,
			y: self.y * rhs.y,
			z: self.z * rhs.z,
		}
	}
}
impl Mul<f64> for Vec3 {
	type Output = Vec3;
	fn mul(self, rhs: f64) -> Self {
		Vec3 {
			x: self.x * rhs,
			y: self.y * rhs,
			z: self.z * rhs,
		}
	}
}
impl Mul<Vec3> for f64{
	type Output = Vec3;
	fn mul(self, rhs: Vec3) -> Vec3{
		rhs * self
	}
}
impl Add for Vec3 {
	type Output = Vec3;

	fn add(self, other: Self) -> Self{
		Vec3 {
			x: self.x + other.x,
			y: self.y + other.y,
			z: self.z + other.z,
		}
	}
}
impl Sub<Vec3> for Vec3 {
	type Output = Vec3;

	fn sub(self, other: Vec3) -> Vec3 {
		Vec3 {
			x: self.x - other.x,
			y: self.y - other.y,
			z: self.z - other.z,
		}
	}
}
impl Sub<&Vec3> for Vec3 {
	type Output = Vec3;
	fn sub(self, other: &Vec3) -> Vec3{
		Vec3 {
			x: self.x - other.x,
			y: self.y - other.y,
			z: self.z - other.z,
		}
	}
}
impl Vec3 {
	fn new(x: f64, y: f64, z: f64) -> Vec3 {
		Vec3 {x: x, y: y, z: z}
	}
	fn unit_vector(v: Vec3) -> Vec3 {
		let len = v.length();
		v / len
	}
	fn unit_vector_from_ref(v: &Self) -> Vec3{
		let len = v.length();
		Vec3 {
			x : v.x / len,
			y : v.y / len,
			z : v.z / len,
		}
	}

	fn length(&self) -> f64 {
		(self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
	}

	fn squared_length(&self) -> f64 {
		self.x * self.x + self.y * self.y + self.z * self.z
	}

	fn r(&self) -> f64 {self.x}
	fn g(&self) -> f64 {self.y}
	fn b(&self) -> f64 {self.z}

	fn make_unit_vector(&mut self){
		let k = 1.0 / self.length();
		self.x *= k;
		self.y *= k;
		self.z *= k;
	}

	fn dot(self, other: Vec3) -> f64 {
		self.x * other.x + self.y * other.y + self.z * other.z
	}

	fn cross(self, other: Vec3) -> Self {
		Vec3 {
			x: (self.y * other.z - self.z * other.y),
			y: (-(self.x * other.z - other.z * self.x)),
			z: (self.x * other.y - self.y * other.x),
		}
	}
}

#[derive(Debug, Copy, Clone)]
struct Ray {
	A: Vec3,
	B: Vec3,
}

impl Ray {
	fn new(a: Vec3, b: Vec3) -> Self {
		Ray {
			A: a,
			B: b,
		}
	}
	fn origin(&self) -> &Vec3{
		&self.A
	}
	fn direction(&self) -> &Vec3{
		&self.B
	}
	fn point_at_parameter(&self, t: f64) -> Vec3{
		self.A + t * self.B
	}
}

fn color(r: Ray) -> Vec3{
	if hit_sphere(&Vec3::new(0.0, 0.0, -1.0), 0.5, &r){
		return Vec3::new(1.0, 0.0, 0.0)
	}
	let unit_direction = Vec3::unit_vector_from_ref(r.direction());
	let t = 0.5 * (unit_direction.y  + 1.0);
	(1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
}
fn hit_sphere(center: &Vec3, radius: f64, r: &Ray) -> bool{
	let oc = *r.origin() - center;
	let a = Vec3::dot(*r.direction(), *r.direction());
	let b = 2.0 * Vec3::dot(oc, *r.direction());
	let c = Vec3::dot(oc, oc) - radius * radius;
	let discriminant = b * b - 4.0 * a * c;
	discriminant > 0.0
}

fn main() {
	let mut file = File::create("out/image.ppm").expect("Hey why can't I write?!");
	let mut output = String::new();
	let nx = 200;
	let ny = 100;
	output += &format!("P3\n");
	output += &format!("{:?} {:?}\n", nx, ny);
	output += &format!("255\n");

	let lower_left_corner = Vec3::new(-2.0, -1.0, -1.0);
	let horizontal = Vec3::new(4.0, 0.0, 0.0);
	let vertical = Vec3::new(0.0, 2.0, 0.0);
	let origin = Vec3::new(0.0, 0.0, 0.0);
	for j2 in 0..=ny-1{
		let j = (ny - 1) - j2;
		for i in 0..=(nx - 1){

			let u = i as f64 / nx as f64;
			let v = j as f64 / ny as f64;

			let r = Ray::new(origin, lower_left_corner + u * horizontal + v * vertical);
			let col = color(r);

			let ir = (255.99 * col[0]) as i64;
			let ig = (255.99 * col[1]) as i64;
			let ib = (255.99 * col[2]) as i64;
			output += &format!("{:?} {:?} {:?}\n", ir, ig, ib );
		}
	}
	file.write_all(&output.into_bytes()).expect("What could the problem be?!");

}
