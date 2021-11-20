// build-pass
// compile-flags: --crate-type lib -Ccodegen-units=1

// This test declares globals by the same name with different types, which
// caused problems because Module::getOrInsertGlobal would return a Constant*
// bitcast instead of a GlobalVariable* that could access linkage/visibility.
// In alt builds with LLVM assertions this would fail:
//
// rustc: /checkout/src/llvm-project/llvm/include/llvm/Support/Casting.h:269:
// typename cast_retty<X, Y *>::ret_type llvm::cast(Y *) [X = llvm::GlobalValue, Y = llvm::Value]:
// Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
//
// In regular builds, the bad cast was UB, like "Invalid LLVMRustVisibility value!"

pub mod before {
    #[no_mangle]
    pub static GLOBAL1: [u8; 1] = [1];
}

pub mod inner {
    extern "C" {
        pub static GLOBAL1: u8;
        pub static GLOBAL2: u8;
    }

    pub fn call() {
        drop(unsafe { (GLOBAL1, GLOBAL2) });
    }
}

pub mod after {
    #[no_mangle]
    pub static GLOBAL2: [u8; 1] = [2];
}
