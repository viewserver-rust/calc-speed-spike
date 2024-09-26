use std::ptr;
use typed_arena::Arena;
use log::info;

// A DynamicArray that uses arena-allocated memory (using a raw pointer to the arena)
pub struct DynamicArray {
    ptr: *mut f32,
    capacity: usize,
    size: usize,
    arena: *const Arena<f32>,  // Raw pointer to the arena
}

impl DynamicArray {
    // Create a new dynamic array with an initial capacity
    pub fn new(arena: *const Arena<f32>, capacity: usize) -> Self {
        let ptr = unsafe { (*arena).alloc_uninitialized(capacity).as_mut_ptr() } as *mut f32;
        DynamicArray {
            ptr ,
            capacity,
            size: 0,
            arena,
        }
    }

    // Push a new value into the array
    pub fn push(&mut self, value: f32) {
        if self.size == self.capacity {
            // Resize logic if needed: allocate new memory from the arena
            self.resize(self.capacity * 2);
        }

        unsafe {
            ptr::write(self.ptr.add(self.size), value);
        }
        self.size += 1;
        println!("Resized to fit: new size = {}", self.size)
    }

    // Get a value from the array
    pub fn get(&self, index: usize) -> f32 {
        if index >= self.size.clone() {
            panic!("Index out of bounds {} >= {}", index, self.size.clone());
        }
        unsafe { *self.ptr.add(index) }
    }
    pub fn set(&mut self, index: usize, value: f32) {
        if index >= self.size {
            // Resize to fit the new index, and ensure capacity is enough
            self.resize_to_fit(index + 1);
            self.size = index + 1; // Update the size to include the new index
            println!("Resized to fit: new size = {}", self.size)
        }

        unsafe {
            *self.ptr.add(index) = value;
        }
    }

    // Helper method to resize the array if the index exceeds the current capacity
    fn resize_to_fit(&mut self, new_size: usize) {
        if new_size > self.capacity {
            let new_capacity = std::cmp::max(new_size, self.capacity * 2);
            self.resize(new_capacity);
        }
    }
    // Resize the array: allocate new memory from the arena and copy data
    pub fn resize(&mut self, new_capacity: usize) {
        // Allocate new memory from the arena
        println!("Resizing array from {} to {}", self.capacity, new_capacity);
        let new_ptr = unsafe { (*self.arena).alloc_uninitialized(new_capacity).as_mut_ptr() } as *mut f32;

        // Copy old data into the new memory
        unsafe {
            ptr::copy_nonoverlapping(self.ptr, new_ptr, self.size);
        }

        // Update the pointer and capacity
        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }

    // Get the length of the array
    pub fn len(&self) -> usize {
        self.size
    }
}