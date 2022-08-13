/*
A program that renders a scene using the raytracing algorithm.
Target is to do it in less than 1000 lines of code

*/
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

namespace util {

template <typename T>
T clamp(T val, T max) {
    return val < 0.0 ? 0.0 : (val > max ? max : val);
}

template <typename T>
T get_random() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<T> d(0.f, 1.f);
    return d(e);
}

template <typename T>
T get_random(float lo, float hi) {
    static std::default_random_engine e;
    static std::uniform_real_distribution<T> d(lo, hi);
    return d(e);
}

template <typename T>
T get_random(double lo, double hi) {
    static std::default_random_engine e;
    static std::uniform_real_distribution<T> d(lo, hi);
    return d(e);
}

}  // namespace util

namespace math {

constexpr double EPSILON = 0.0000001;

class vec3 {
   public:
    vec3() : e{0, 0, 0} {}

    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    [[nodiscard]] double operator[](int i) const { return e[i]; }

    [[nodiscard]] double &operator[](int i) { return e[i]; }

    vec3 &operator+=(const vec3 &v) noexcept {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3 &operator-=(const vec3 &v) noexcept {
        e[0] -= v.e[0];
        e[1] -= v.e[1];
        e[2] -= v.e[2];
        return *this;
    }

    vec3 &operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3 &operator/=(const double t) {
        const auto recip = 1 / t;
        e[0] *= recip;
        e[1] *= recip;
        e[2] *= recip;
        return *this;
    }

    vec3 operator+(const vec3 &v) const noexcept { return {e[0] + v.e[0], e[1] + v.e[1], e[2] + v.e[2]}; }

    vec3 operator-(const vec3 &v) const noexcept { return {e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]}; }

    vec3 operator*(const vec3 &v) const noexcept { return {e[0] * v.e[0], e[1] * v.e[1], e[2] * v.e[2]}; }

    friend vec3 operator*(double t, const vec3 &v) { return {t * v.e[0], t * v.e[1], t * v.e[2]}; }

    vec3 operator*(double t) const noexcept { return t * *this; }

    friend vec3 operator/(double lhs, const vec3 &rhs) { return {lhs / rhs.e[0], lhs / rhs.e[1], lhs / rhs.e[2]}; }

    vec3 operator/(double b) const noexcept {
        const auto reciprocal = 1.0 / b;
        return {e[0] * reciprocal, e[1] * reciprocal, e[2] * reciprocal};
    }

    vec3 operator-() const { return {-e[0], -e[1], -e[2]}; }

    [[nodiscard]] double length() const noexcept { return sqrt(length_squared()); }

    [[nodiscard]] double length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    vec3 &normalized() { return *this /= length(); }

    [[nodiscard]] constexpr double x() const noexcept { return e[0]; }

    [[nodiscard]] constexpr double y() const noexcept { return e[1]; }

    [[nodiscard]] constexpr double z() const noexcept { return e[2]; }

    [[nodiscard]] double &x() noexcept { return e[0]; }

    [[nodiscard]] double &y() noexcept { return e[1]; }

    [[nodiscard]] double &z() noexcept { return e[2]; }

    [[nodiscard]] static vec3 xAxis() { return {1, 0, 0}; }

    [[nodiscard]] static vec3 yAxis() { return {0, 1, 0}; }

    [[nodiscard]] static vec3 zAxis() { return {0, 0, 1}; }

    [[nodiscard]] bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    static vec3 random() {
        return {util::get_random<double>(), util::get_random<double>(), util::get_random<double>()};
    }

    static vec3 random(double lo, double hi) {
        return {util::get_random<double>(lo, hi), util::get_random<double>(lo, hi),
                util::get_random<double>(lo, hi)};
    }

   public:
    double e[3];
};

// Type aliases for vec3
using point3 = vec3;  // 3D point
using color = vec3;   // RGB color

// vec3 Utility Functions
inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3 &u, const vec3 &v) {
    return {u.e[1] * v.e[2] - u.e[2] * v.e[1], u.e[2] * v.e[0] - u.e[0] * v.e[2],
            u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}

inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

inline vec3 random_in_unit_disk() {
    while (true) {
        auto p = vec3(util::get_random<double>(-1.f, 1.f), util::get_random<double>(-1.f, 1.f), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

inline vec3 random_in_unit_sphere() {
    while (true) {
        auto p = vec3::random(-1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

inline vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

inline vec3 random_in_hemisphere(const vec3 &normal) {
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0)  // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

inline vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

inline std::pair<int, int> get_pixel_coordinates(int pix, int width, int height) {
    return {pix % width, pix / width};
}

class ray {
   public:
    ray() = default;

    ray(const point3 &origin, const vec3 &direction)
        : orig(origin), dir(direction), tm(0) {}

    ray(const point3 &origin, const vec3 &direction, double time)
        : orig(origin), dir(direction), tm(time) {}

    [[nodiscard]] point3 origin() const { return orig; }

    [[nodiscard]] vec3 direction() const { return dir; }

    [[nodiscard]] double time() const { return tm; }

    [[nodiscard]] point3 at(double t) const {
        return orig + t * dir;
    }

   public:
    point3 orig;
    vec3 dir;
    double tm{};
};
}  // namespace math

using math::color;
using math::point3;
using math::ray;
using math::vec3;

class material;

struct intersection_record {
    intersection_record() = default;

    intersection_record(const point3 &p, const vec3 &n, material *mat, double t, double u, double v, bool is_front_face)
        : p(p), normal(n), mat(mat), t(t), u(u), v(v), is_front_face(is_front_face) {}

    point3 p;
    vec3 normal;
    material *mat{};
    double t{}, u{}, v{};
    bool is_front_face{};
};

struct camera {
    camera(point3 lookfrom, point3 lookat, vec3 vup, double vfov, double aspect, double aperture, double focus_dist,
           double t0, double t1) {
        auto theta = vfov * M_PI / 180.0;
        auto half_height = tan(theta / 2);
        auto half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
        lens_radius = aperture / 2;
        time0 = t0;
        time1 = t1;
    }

    [[nodiscard]] ray get_ray(double s, double t) const {
        auto rd = lens_radius * math::random_in_unit_disk();
        auto offset = u * rd.x() + v * rd.y();
        return {origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset,
                time0 + util::get_random<double>(time0, time1)};
    }

    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    double lens_radius;
    double time0, time1;
};

class texture {
   public:
    [[nodiscard]] virtual color value(double u, double v, const vec3 &p) const = 0;
};

class solid_color : public texture {
   public:
    solid_color() = default;

    explicit solid_color(color c) : color_value(c) {}

    solid_color(double red, double green, double blue)
        : solid_color(color(red, green, blue)) {}

    [[nodiscard]] color value(double u, double v, const vec3 &p) const override {
        return color_value;
    }

   private:
    color color_value;
};

class checker_texture : public texture {
   public:
    checker_texture() = default;

    checker_texture(color c1, color c2) : even(c1), odd(c2) {}

    [[nodiscard]] color value(double u, double v, const vec3 &p) const override {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0)
            return odd;
        else
            return even;
    }

   public:
    color odd, even;
};

class material {
   public:
    [[nodiscard]] virtual color emitted(double u, double v, const point3 &p) const {
        return {0, 0, 0};
    }

    virtual bool scatter(
        const ray &r_in, const intersection_record &rec, color &attenuation, ray &scattered) const = 0;
};

class lambertian : public material {
   public:
    explicit lambertian(const color &a) : albedo(new solid_color(a)) {}

    explicit lambertian(texture *a) : albedo(a) {}

    ~lambertian() { delete albedo; }

    bool scatter(
        const ray &r_in, const intersection_record &rec, color &attenuation, ray &scattered) const override {
        auto scatter_direction = rec.normal + math::random_unit_vector();

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

   public:
    texture *albedo;
};

class metal : public material {
   public:
    metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    bool scatter(
        const ray &r_in, const intersection_record &rec, color &attenuation, ray &scattered) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * math::random_in_unit_sphere(), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

   public:
    color albedo;
    double fuzz;
};

class dielectric : public material {
   public:
    explicit dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    bool scatter(
        const ray &r_in, const intersection_record &rec, color &attenuation, ray &scattered) const override {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.is_front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > util::get_random<double>())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction, r_in.time());
        return true;
    }

   public:
    double ir;  // Index of Refraction

   private:
    static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

class diffuse_light : public material {
   public:
    explicit diffuse_light(texture *a) : emit(a) {}

    explicit diffuse_light(color c) : emit(new solid_color(c)) {}

    ~diffuse_light() { delete emit; }

    bool scatter(
        const ray &r_in, const intersection_record &rec, color &attenuation, ray &scattered) const override {
        return false;
    }

    [[nodiscard]] color emitted(double u, double v, const point3 &p) const override {
        return emit->value(u, v, p);
    }

   public:
    texture *emit;
};

class aabb {
   public:
    aabb() = default;

    aabb(const point3 &a, const point3 &b) : _min(a), _max(b) {}

    explicit aabb(const point3 &c) : _min(c), _max(c) {}

    aabb(double x0, double x1, double y0, double y1, double z0, double z1)
        : _min(x0, y0, z0), _max(x1, y1, z1) {}

    aabb(const aabb &other) = default;

    aabb &operator=(const aabb &other) = default;

   private:
    point3 _min, _max;
};

class intersectable {
   public:
    virtual bool intersect(const ray &r, intersection_record &rec) const = 0;

    virtual bool bounding_box(double t0, double t1, point3 &bbox_min, point3 &bbox_max) const = 0;
};

class sphere : public intersectable {
   public:
    sphere() = default;

    sphere(const point3 &c, double r, material *m) : center(c), radius(r), mat(m) {}

    bool intersect(const ray &r, intersection_record &rec) const override {
        vec3 oc = r.origin() - center;
        auto a = dot(r.direction(), r.direction());
        auto b = dot(oc, r.direction());
        auto c = dot(oc, oc) - radius * radius;
        auto discriminant = b * b - a * c;
        if (discriminant > 0) {
            auto temp = (-b - sqrt(discriminant)) / a;
            if (temp < 0)
                temp = (-b + sqrt(discriminant)) / a;
            if (temp < 0)
                return false;
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat = mat;
            rec.u = (atan2(rec.p.z() - center.z(), rec.p.x() - center.x()) + M_PI) / (2.0 * M_PI);
            rec.v = (acos(rec.p.y() - center.y()) + M_PI) / (2.0 * M_PI);
            return true;
        }
        return false;
    }

    bool bounding_box(double t0, double t1, point3 &bbox_min, point3 &bbox_max) const override {
        bbox_min = center - vec3(radius, radius, radius);
        bbox_max = center + vec3(radius, radius, radius);
        return true;
    }

   public:
    point3 center;
    double radius{};
    material *mat{};
};

class bvh_node : public intersectable {
   public:
    bvh_node() = default;

    bvh_node(std::vector<intersectable *> &objects, int start, int end, int axis) {
        int axis_index = (axis + 1) % 3;
        int axis_index2 = (axis + 2) % 3;
        int mid = (start + end) / 2;
        int min_index = start;
        int max_index = start;
        for (int i = start; i < end; i++) {
            if (objects[i]->bounding_box(0, 0, bbox_min, bbox_max)) {
                if (bbox_min[axis] < bbox_min[axis_index])
                    min_index = i;
                if (bbox_max[axis] > bbox_max[axis_index])
                    max_index = i;
            }
        }
        left = objects[min_index];
        right = nullptr;
    }

    bool intersect(const ray &r, intersection_record &rec) const override {
        if (left->intersect(r, rec)) {
            intersection_record temp_rec;
            if (right && right->intersect(r, temp_rec)) {
                if (temp_rec.t < rec.t)
                    rec = temp_rec;
            }
            return true;
        }
        if (right && right->intersect(r, rec))
            return true;
        return false;
    }

    bool bounding_box(double t0, double t1, point3 &box_min, point3 &box_max) const override {
        box_min = bbox_min;
        box_max = bbox_max;
        return true;
    }

   public:
    point3 bbox_min, bbox_max;
    intersectable *left{}, *right{};
};

// moller trumbore algorithm
bool triangle_intersect(const std::array<point3, 3> &v, const std::array<vec3, 3> &n, const ray &r,
                        intersection_record &rec) {
    vec3 e1 = v[1] - v[0];
    vec3 e2 = v[2] - v[0];
    vec3 s1 = cross(r.direction(), e2);
    double divisor = dot(s1, e1);
    if (divisor > -math::EPSILON && divisor < math::EPSILON)
        return false;
    double inv_divisor = 1.0 / divisor;
    vec3 s = r.origin() - v[0];
    double b1 = dot(s, s1) * inv_divisor;
    if (b1 < 0.0 || b1 > 1.0)
        return false;
    vec3 s2 = cross(s, e1);
    double b2 = dot(r.direction(), s2) * inv_divisor;
    if (b2 < 0.0 || b1 + b2 > 1.0)
        return false;
    double t = dot(e2, s2) * inv_divisor;
    if (t < math::EPSILON)
        return false;
    rec.t = t;
    rec.p = r.at(t);
    rec.normal = n[0] + b1 * (n[1] - n[0]) + b2 * (n[2] - n[0]);
    rec.normal.normalized();
    return true;
}

class scene {
    using triangle_vert = std::array<point3, 3>;
    using triangle_normals = std::array<vec3, 3>;

    std::vector<triangle_vert> t_vertices;
    std::vector<triangle_normals> t_normals;
    std::vector<material *> t_material;

    std::vector<sphere> spheres;
    std::vector<material *> s_material;

   public:
    void add_triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2, material *material) {
        t_vertices.push_back({v0, v1, v2});
        t_normals.push_back({(v1 - v0).normalized(), (v2 - v0).normalized(), (v0 - v1).normalized()});
        t_material.push_back(material);
    }

    void add_sphere(const vec3 &center, double radius, material *material) {
        spheres.emplace_back(center, radius, material);
        s_material.push_back(material);
    }

    intersection_record intersect_spheres(const ray &r, double near) {
        intersection_record closest_rec;
        closest_rec.t = std::numeric_limits<double>::max();
        for (auto &s : spheres) {
            intersection_record rec;
            if (s.intersect(r, rec) && rec.t < closest_rec.t && rec.t > near) {
                closest_rec = rec;
            }
        }
        return closest_rec;
    }

    intersection_record intersect_triangles(const ray &r, double near) {
        intersection_record closest_rec;
        closest_rec.t = std::numeric_limits<double>::max();
        for (size_t i = 0; i < t_vertices.size(); i++) {
            intersection_record rec;
            if (triangle_intersect(t_vertices[i], t_normals[i], r, rec) && rec.t < closest_rec.t && rec.t > near) {
                closest_rec = rec;
                closest_rec.mat = t_material[i];
            }
        }
        return closest_rec;
    }

    intersection_record intersect(const ray &r) {
        intersection_record closest_so_far = intersection_record();
        closest_so_far.t = std::numeric_limits<double>::max();
        intersection_record rec = intersect_spheres(r, closest_so_far.t);
        if (rec.t < closest_so_far.t) {
            closest_so_far = rec;
        }
        rec = intersect_triangles(r, closest_so_far.t);
        if (rec.t < closest_so_far.t) {
            closest_so_far = rec;
        }
        return closest_so_far;
    }
};

namespace scenes {

// create a cornell box scene
// with a sphere and a piramid and a cube in the center
scene create_scene_box(const camera &cam, const color &bg_color) {
    scene s;
    // create the cornell box
    // s.add_triangle({-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, new lambertian(color(0.5, 0.5, 0.5)));

    // create the sphere
    s.add_sphere({0, 0, -1}, 0.5, new lambertian(color(0.1, 0.2, 0.5)));

    // create a pyramid

    // create a cube

    return s;
}

};  // namespace scenes

int main() {
    const auto [width, height] = std::make_pair(512u, 512u);

    const int samples = 100;
    const int max_depth = 50;
    const auto [camera_pos, camera_dir] = std::make_pair(vec3{0.f, 0.f, 0.f}, vec3{0.f, 0.f, -1.f});

    const vec3 up{0.f, 1.f, 0.f};

    const auto [aperture, focus_dist] = std::make_pair(0.f, 1.f);
    const auto [fov, aspect_ratio] = std::make_pair(60.f, width / static_cast<float>(height));

    camera cam{camera_pos, camera_dir, up, fov, aspect_ratio, aperture, focus_dist, 0.f, 1.f};
    auto world = scenes::create_scene_box(cam, color{0.f, 0.f, 0.f});
    // std::ofstream output;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<color> framebuffer(width * height);

#pragma omp parallel for
    for (int pix = 0u; pix < width * height; pix++) {
        for (int s = 0u; s < samples; s++) {
            auto [u, v] = math::get_pixel_coordinates(pix, width, height);
            auto r = cam.get_ray(u, v);
            auto rec = world.intersect(r);
            framebuffer[pix] += color{rec.p.x(), rec.p.y(), rec.p.z()};
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time:" << duration.count() << " milliseconds\n";
    std::cout << "Rays:" << width * height * samples << '\n';
    std::cout << "Rays/ms:" << (width * height * samples) / duration.count() << '\n';

    std::ofstream ofs("./out.ppm", std::ios::binary);
    ofs << "P3\n"
        << width << ' ' << height << '\n'
        << 255 << '\n';

    for (const color &c : framebuffer) {
        auto r = c.x();
        auto g = c.y();
        auto b = c.z();

        if (r != r) r = 0.0;
        if (g != g) g = 0.0;
        if (b != b) b = 0.0;

        auto scale = 1.0 / samples;
        r = sqrt(scale * r);
        g = sqrt(scale * g);
        b = sqrt(scale * b);

        ofs << static_cast<int>(256 * util::clamp(r, 0.999)) << ' '
            << static_cast<int>(256 * util::clamp(g, 0.999)) << ' '
            << static_cast<int>(256 * util::clamp(b, 0.999)) << std::endl;
    }

    return 0;
}