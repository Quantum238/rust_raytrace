extern crate rand;
use std::fs::File;
use std::io::prelude::*;
use std::ops::{Add, Sub, Index, AddAssign, SubAssign, Mul, Div, MulAssign, DivAssign};
// use std::path::Path;
use std::fmt::Debug;
use rand::Rng;

trait Material<T>: Debug where T: Material<T>{
	fn scatter(r_in: Ray, attenuation: Vec3, scattered: Ray, hit_record: HitRecord) -> Option<Vec3>;
}

#[derive(Debug)]
struct Lambertian {
	albedo: Vec3,
}
impl Lambertian {
	fn new(a: Vec3) -> Self{
		Lambertian {albedo: a}
	}
	// add code here
}
impl Material for Lambertian{
	fn scatter(&self, r_in: Ray, attenuation: Vec3, scattered: Ray, hit_record: HitRecord) -> Option<Vec3>{
		let target = hit_record.p + hit_record.normal + random_in_unit_sphere();
		let scattered = Ray::new(target - hit_record.p);
		Some(self.albedo)
		
	}
}

#[derive(Debug)]
struct Metal {
	albedo: Vec3,
}
impl Material for Metal{
	fn scatter(&self, r_in: Ray, attenuation: Vec3, scattered: Ray, hit_record: HitRecord) -> bool{
		let reflected = Vec3::reflect(Vec3::unit_vector(r_in.direction()), hit_record.normal);
		let scattered = Ray::new(hit_record.p, reflected);
		attenuation = albedo;
		Vec3::dot(scattered.direction(), hit_record.normal) > 0
	}
}

#[derive(Debug, Copy, Clone)]
struct Camera{
	lower_left_corner: Vec3,
	horizontal: Vec3,
	vertical: Vec3,
	origin: Vec3,
}
impl Camera{
	fn new() -> Self{
		Camera {
			lower_left_corner: Vec3::new(-2.0, -1.0, -1.0),
			horizontal: Vec3::new(4.0, 0.0, 0.0),
			vertical: Vec3::new(0.0, 2.0, 0.0),
			origin: Vec3::new(0.0, 0.0, 0.0),
		}
	}
	fn get_ray(&self, u: f64, v: f64) -> Ray{
		Ray::new(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical - self.origin)
	}
}

#[derive(Debug, Copy, Clone)]
struct HitRecord<T: Material<T>>{
	t: f64,
	p: Vec3,
	normal: Vec3,
	material: T,

}
impl<T: Material<T>> HitRecord<T>{
	fn new(t: f64, p: Vec3, normal: Vec3, mat: T) -> Self{
		HitRecord {t: t, p: p, normal: normal, material: mat,}
	}
	fn new_from_empty() -> Self{
		HitRecord {t: 0.0, p: Vec3::new(0.0, 0.0, 0.0), normal:Vec3::new(0.0, 0.0, 0.0)}
	}
}
trait Hitable<T>: Debug where T: Material<T> {
	fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<T>>;
}

#[derive(Debug, Copy, Clone)]
struct Sphere{
	center: Vec3,
	radius: f64,
}
impl Sphere {
	fn new(cen: Vec3, r: f64) -> Self{
		Sphere{ center: cen, radius: r}
	}
}
impl<T> Hitable<T> for Sphere where T: Material<T>{
	fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<T>>{
		let oc = *r.origin() - self.center;
		let a = Vec3::dot(*r.direction(), *r.direction());
		let b = Vec3::dot(oc, *r.direction());
		let c = Vec3::dot(oc, oc) - self.radius * self.radius;
		let mut record = HitRecord::new_from_empty();

		let discriminant = b * b - a * c;
		if discriminant > 0.0{
			let root = (-b - discriminant.sqrt()) / a;
			if root < t_max && root > t_min {
				record.t = root;
				record.p = r.point_at_parameter(record.t);
				record.normal = (record.p - self.center) / self.radius;
				return Some(record);
			}
			let root = (-b + discriminant.sqrt()) / a;
			if root < t_max && root > t_min{
				record.t = root;
				record.p = r.point_at_parameter(record.t);
				record.normal = (record.p - self.center) / self.radius;
				return Some(record);
			}
		}
		None
	}
}

#[derive(Debug)]
struct HitableList<T: Hitable<T>> where T: Material<T>{
	list: Vec<T>,
	list_size: usize
}
impl<T: Hitable<T>> HitableList<T> where T: Material<T>{
	fn new(l: Vec<T>, n: usize) -> Self{
		HitableList {
			list: l,
			list_size: n
		}
	}
}
impl<T: Hitable<T>> Hitable<T> for HitableList<T> where T: Material<T>{
	fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord<T>>{
		let mut hit_anything = false;
		let mut closest_so_far = t_max;
		let mut record = HitRecord::new_from_empty();
		for i in 0..self.list_size{
			let entry_hit_record = self.list[i].hit(r, t_min, closest_so_far);
			match entry_hit_record {
				Some(hit_record) => {
					hit_anything = true;
					closest_so_far = hit_record.t;
					record.t = hit_record.t;
					record.p = hit_record.p;
					record.normal = hit_record.normal;
				},
				None => {},
			};
		}
		if hit_anything{
			return Some(record);
		}
		None
	}
}

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

	fn reflect(v: Self, n: Self) -> Self{
	v - 2 * Vec3::dot(v, n) * n
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
fn random_in_unit_sphere() -> Vec3{
	let mut p = Vec3::new(10.0, 10.0, 10.0);
	let mut rng = rand::thread_rng();
	while p.squared_length() >= 1.0{
		let rand1: f64 = rng.gen();
		let rand2: f64 = rng.gen();
		let rand3: f64 = rng.gen();
		p = 2.0 * Vec3::new(rand1, rand2, rand3) - Vec3::new(1.0,1.0,1.0);
	}
	p
}
fn color<T: Hitable<T>>(r: Ray, world: &T) -> Vec3 where T: Material<T>{
	let world_hit = world.hit(&r, 0.001, std::f64::MAX);
	match world_hit {
		Some(hit_record) => {
			let target = hit_record.p + hit_record.normal + random_in_unit_sphere();
			return 0.5 * color(Ray::new(hit_record.p, target - hit_record.p), world);
		}
		None => {
		},
	};

	let unit_direction = Vec3::unit_vector_from_ref(r.direction());
	let t = 0.5 * (unit_direction.y  + 1.0);
	(1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
}

fn main() {
	let mut file = File::create("out/image.ppm").expect("Hey why can't I write?!");
	let mut output = String::new();

	let nx = 200;
	let ny = 100;
	let ns = 100;

	output += &format!("P3\n");
	output += &format!("{:?} {:?}\n", nx, ny);
	output += &format!("255\n");

	let lower_left_corner = Vec3::new(-2.0, -1.0, -1.0);
	let horizontal = Vec3::new(4.0, 0.0, 0.0);
	let vertical = Vec3::new(0.0, 2.0, 0.0);
	let origin = Vec3::new(0.0, 0.0, 0.0);

	let list = vec![
		Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5),
		Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0)
	];
	let world = HitableList::new(list, 2);
	let cam = Camera::new();

	let mut rng = rand::thread_rng();

	for j2 in 0..=ny-1{
		let j = (ny - 1) - j2;
		for i in 0..=(nx - 1){
			println!("{:?}, {:?}",i, j );
			let mut col = Vec3::new(0.0, 0.0, 0.0);
			for s in 0..ns{
				let rand1: f64 = rng.gen();
				let rand2: f64 = rng.gen();
				let u = (i as f64 + rand1)/ nx as f64;
				let v = (j as f64 + rand2) / ny as f64;
				let r = cam.get_ray(u, v);
				let p = r.point_at_parameter(2.0);
				col += color(r, &world);
			}
			col /= ns as f64;
			col = Vec3::new(col[0].sqrt(), col[1].sqrt(), col[2].sqrt());

			let ir = (255.99 * col[0]) as i64;
			let ig = (255.99 * col[1]) as i64;
			let ib = (255.99 * col[2]) as i64;
			output += &format!("{:?} {:?} {:?}\n", ir, ig, ib );
		}
	}
	file.write_all(&output.into_bytes()).expect("What could the problem be?!");

}
