extern crate rand;
use std::fs::File;
use std::io::prelude::*;
use std::ops::{Add, Sub, Index, AddAssign, SubAssign, Mul, Div, MulAssign, DivAssign};
use std::fmt::Debug;
use rand::Rng;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
enum Material {
	Lambertian(Vec3),
	Metal(Vec3, f64),
	Dielectric(f64),
}

impl Material {
	fn scatter(&self, r_in: Ray, hit_record: HitRecord) -> Option<(Vec3, Ray)>{
		match self {
			Material::Lambertian(albedo) => {
				let target = hit_record.p + hit_record.normal + random_in_unit_sphere();
				let scattered = Ray::new(hit_record.p, target - hit_record.p);
				Some((*albedo, scattered))
			},
			Material::Metal(albedo, possible_fuzz) => {
				let fuzz = if possible_fuzz < &1.0 {*possible_fuzz} else {1.0};
				let reflected = Vec3::reflect(Vec3::unit_vector(*r_in.direction()), hit_record.normal);
				let scattered = Ray::new(hit_record.p, reflected + fuzz * random_in_unit_sphere());
				if Vec3::dot(*scattered.direction(), hit_record.normal) > 0.0{
					return Some((*albedo, scattered));
				}
				None
			},
			Material::Dielectric(ref_index) => {
				let reflected = Vec3::reflect(*r_in.direction(), hit_record.normal);
				let attenuation = Vec3::new(1.0, 1.0, 1.0);
				let mut outward_normal = Vec3::new(0.0, 0.0, 0.0);
				let mut ni_over_nt = 0.0;
				let mut cosine = 0.0;
				let mut reflect_prob = 0.0;
				if Vec3::dot(*r_in.direction(), hit_record.normal) > 0.0{
					outward_normal = -1.0 * hit_record.normal;
					ni_over_nt = *ref_index;
					cosine = ref_index * Vec3::dot(*r_in.direction(), hit_record.normal) / r_in.direction().length();
				} else {
					outward_normal = hit_record.normal;
					ni_over_nt = 1.0 / *ref_index;
					cosine = -1.0 * Vec3::dot(*r_in.direction(), hit_record.normal) / r_in.direction().length();
				}

				let mut scattered = Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
				let mut outer_refracted = Vec3::new(0.0, 0.0, 0.0);
				match self.refract(*r_in.direction(), outward_normal, ni_over_nt) {
					Some(refracted) => {
						// scattered = Ray::new(hit_record.p, refracted);
						reflect_prob = schlick(cosine, *ref_index);
						outer_refracted = refracted;
					},
					None => {
						// scattered = Ray::new(hit_record.p, reflected);
						reflect_prob = 1.0;
					},
				};
				let mut rng = rand::thread_rng();
				let rand1: f64 = rng.gen();
				if rand1 < reflect_prob{
					scattered = Ray::new(hit_record.p, reflected);
				}else{
					scattered = Ray::new(hit_record.p, outer_refracted);
				}
				return Some((attenuation, scattered));
				// return Some((scattered, Ray::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0))));
			}
		}
	}

	fn refract(&self, v: Vec3, n: Vec3, ni_over_nt: f64) -> Option<Vec3>{
		let uv = Vec3::unit_vector(v);
		let dt = Vec3::dot(uv, n);
		let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
		if discriminant > 0.0{
			return Some(ni_over_nt * (uv - n*dt) - n * discriminant.sqrt());
		}
		None
	}
}
fn schlick(cosine: f64, ref_index: f64) -> f64{
	let r0 = (1.0 - ref_index) / (1.0 + ref_index);
	let r0 = r0 * r0;
	r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)

}
// impl Hitable for Material{
// 	fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>{
// 		self.hit(r, t_min, t_max)
// 	}

// }

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
struct HitRecord{
	t: f64,
	p: Vec3,
	normal: Vec3,
	material: Material,

}
impl HitRecord{
	fn new(t: f64, p: Vec3, normal: Vec3, mat: Material) -> Self{
		HitRecord {
			t: t,
			p: p,
			normal: normal,
			material: mat,
		}
	}
	fn new_from_empty() -> Self{
		HitRecord {
			t: 0.0,
			p: Vec3::new(0.0, 0.0, 0.0),
			normal:Vec3::new(0.0, 0.0, 0.0),
			material: Material::Lambertian(Vec3::new(0.0, 0.0, 0.0)),
		}
	}
}
trait Hitable: Debug {
	fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
	fn get_material(&self) -> Material;
}

#[derive(Debug, Copy, Clone)]
struct Sphere{
	center: Vec3,
	radius: f64,
	material: Material,
}
impl Sphere {
	fn new(cen: Vec3, r: f64, mat: Material) -> Self{
		Sphere{ center: cen, radius: r, material: mat}
	}
}
impl Hitable for Sphere{
	fn get_material(&self) -> Material{
		self.material
	}
	fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>{
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
struct HitableList<T: Hitable>{
	list: Vec<T>,
	list_size: usize
}
impl<T: Hitable> HitableList<T>{
	fn new(l: Vec<T>, n: usize) -> Self{
		HitableList {
			list: l,
			list_size: n
		}
	}
}
impl<T: Hitable> Hitable for HitableList<T> {
	fn get_material(&self) -> Material{
		Material::Lambertian(Vec3::new(0.0, 0.0, 0.0))
	}
	fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord>{
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
					record.material = self.list[i].get_material();
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
	v - 2.0 * Vec3::dot(v, n) * n
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
fn random_in_unit_disk() -> Vec3{
	let mut p = Vec3::new(10.0, 10.0, 10.0);
	let mut rng = rand::thread_rng();
	while Vec3::dot(p,p) >= 1.0 {
		let rand1 = rng.gen();
		let rand2 = rng.gen();
		p = 2.0 * Vec3::new(rand1, rand2, 0.0) - Vec3::new(1.0, 1.0, 0.0);
	}
	p
}
fn color<T>(r: Ray, world: &HitableList<T>, depth: i32) -> Vec3 where T: Hitable{
	let world_hit = world.hit(&r, 0.001, std::f64::MAX);
	match world_hit {
		Some(hit_record) => {
			let possible_scatter = hit_record.material.scatter(r, hit_record);
			match (depth < 50, possible_scatter) {
				(true, Some((attenuation, scattered_ray))) => {
					return attenuation * color(scattered_ray, world, depth + 1);
				},
				_ => return Vec3::new(0.0, 0.0, 0.0),
			};
		}
		None => {
			let unit_direction = Vec3::unit_vector_from_ref(r.direction());
			let t = 0.5 * (unit_direction.y  + 1.0);
			(1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
		},
	}

}

fn get_random_scene() -> HitableList<Sphere>{
	let n = 500;
	let mut list = vec![
		Sphere::new(
			Vec3::new(0.0, -1000.0, 0.0),
			1000.0,
			Material::Lambertian(Vec3::new(0.5, 0.5, 0.5))
		),
	];
	let mut rng = rand::thread_rng();
	let mut i = 1;
	for a in -11..11{
		for b in -11..11{
			let choose_material: f64 = rng.gen();
			let rand1: f64 = rng.gen();
			let rand2: f64 = rng.gen();
			let center = Vec3::new(a as f64 + 0.9 * rand1, 0.2, b as f64 + 0.9 * rand2);
			if (center - Vec3::new(4.0, 0.2, 0.0)).length() > 0.9{
				if choose_material < 0.8{
					let rand3: f64 = rng.gen();
					let rand4: f64 = rng.gen();
					let rand5: f64 = rng.gen();
					let rand6: f64 = rng.gen();
					let rand7: f64 = rng.gen();
					let rand8: f64 = rng.gen();
					list.push(Sphere::new(
						center,
						0.2,
						Material::Lambertian(
							Vec3::new(
								rand3 * rand4,
								rand5 * rand6,
								rand7 * rand8
							)
						)
					));
				} else if choose_material < 0.95{
					let rand9: f64 = rng.gen();
					let rand10: f64 = rng.gen();
					let rand11: f64 = rng.gen();
					let rand12: f64 = rng.gen();
					list.push(Sphere::new(
						center,
						0.2,
						Material::Metal(
							Vec3::new(
								0.5 * (1.0 + rand9),
								0.5 * (1.0 + rand10),
								0.5 * (1.0 + rand11)
							), 
							0.5 * rand12
						)
					));
				} else {
					list.push(Sphere::new(center, 0.2, Material::Dielectric(1.5)));
				}

				}
			}
		}
		// list.push(Sphere::new(Vec3::new(0.0, 1.0, 0.0), 1.0, Material::Dielectric(1.5)));
		// list.push(Sphere::new(Vec3::new(-4.0, 1.0, 0.0), 1.0, Material::Lambertian(Vec3::new(0.4, 0.2, 0.1))));
		// list.push(Sphere::new(Vec3::new(4.0, 1.0, 0.0), 1.0, Material::Metal(Vec3::new(0.7, 0.6, 0.5), 0.0)));

		let len = list.len();
		HitableList::new(list, len)
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
	let R = (PI / 4.0).cos();
	let list = vec![
		// Sphere::new(Vec3::new(-R, 0.0, -1.0), R, Material::Lambertian(Vec3::new(0.0, 0.0, 1.0))),
		// Sphere::new(Vec3::new(R, 0.0, -1.0), R, Material::Lambertian(Vec3::new(1.0, 0.0, 0.0))),
		Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5, Material::Lambertian(Vec3::new(0.1, 0.2, 0.5))),
		Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0, Material::Lambertian(Vec3::new(0.8, 0.8, 0.0))),
		Sphere::new(Vec3::new(1.0, 0.0, -1.0), 0.5, Material::Metal(Vec3::new(0.8, 0.6, 0.2), 0.3)),
		Sphere::new(Vec3::new(-1.0, 0.0, -1.0), 0.5, Material::Dielectric(1.5)),
		Sphere::new(Vec3::new(-1.0, 0.0, -1.0), -0.45, Material::Dielectric(1.5)),
	];
	// let world = HitableList::new(list, 5);
	let world = get_random_scene();
	let cam = Camera::new();

	let mut rng = rand::thread_rng();

	for j2 in 0..=ny-1{
		let j = (ny - 1) - j2;
		for i in 0..=(nx - 1){
			let mut col = Vec3::new(0.0, 0.0, 0.0);
			for _s in 0..ns{
				let rand1: f64 = rng.gen();
				let rand2: f64 = rng.gen();
				let u = (i as f64 + rand1)/ nx as f64;
				let v = (j as f64 + rand2) / ny as f64;
				let r = cam.get_ray(u, v);
				col += color(r, &world, 0);
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
