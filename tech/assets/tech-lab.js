(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.TechLab = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  const calculatorScope = {
    sin: Math.sin,
    cos: Math.cos,
    tan: Math.tan,
    asin: Math.asin,
    acos: Math.acos,
    atan: Math.atan,
    sqrt: Math.sqrt,
    pow: Math.pow,
    log: Math.log10,
    ln: Math.log,
    abs: Math.abs,
    PI: Math.PI,
    E: Math.E,
  };

  const allowedCalculatorNames = new Set(Object.keys(calculatorScope));
  const allowedCalculatorPattern = /^[0-9+\-*/^().,\sA-Za-z]+$/;

  function normalizeNumber(value) {
    if (Number.isInteger(value)) {
      return value;
    }
    return Number(value.toFixed(10));
  }

  function evaluateScientificExpression(expression) {
    const raw = String(expression || "").trim();
    if (!raw) {
      throw new Error("Enter an expression first.");
    }

    if (!allowedCalculatorPattern.test(raw)) {
      throw new Error("Only calculator symbols are allowed.");
    }

    const identifiers = raw.match(/[A-Za-z_]+/g) || [];
    for (const identifier of identifiers) {
      if (!allowedCalculatorNames.has(identifier)) {
        throw new Error("Unsupported function in expression.");
      }
    }

    const normalized = raw.replace(/\^/g, "**");
    const evaluator = new Function(
      ...Object.keys(calculatorScope),
      '"use strict"; return (' + normalized + ');'
    );
    const result = evaluator(...Object.values(calculatorScope));

    if (!Number.isFinite(result)) {
      throw new Error("Expression returned a non-finite result.");
    }

    return normalizeNumber(result);
  }

  function sanitizeText(value) {
    return String(value || "").trim();
  }

  function normalizeEmail(value) {
    return sanitizeText(value).toLowerCase();
  }

  function validateRegistration(form) {
    const fullName = sanitizeText(form.fullName);
    const email = normalizeEmail(form.email);
    const city = sanitizeText(form.city);
    const password = String(form.password || "");
    const confirmPassword = String(form.confirmPassword || "");
    const errors = [];

    if (!fullName) {
      errors.push("Full name is required.");
    }
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      errors.push("Enter a valid email address.");
    }
    if (!city) {
      errors.push("City is required.");
    }
    if (password.length < 8) {
      errors.push("Password must be at least 8 characters.");
    }
    if (password !== confirmPassword) {
      errors.push("Password confirmation does not match.");
    }

    return {
      isValid: errors.length === 0,
      errors: errors,
      normalized: {
        fullName: fullName,
        email: email,
        city: city,
        password: password,
      },
    };
  }

  function createSeedUsers() {
    return [
      {
        fullName: "Demo Analyst",
        email: "demo@decision.local",
        city: "Pune",
        password: "Demo1234!",
      },
    ];
  }

  function registerUser(users, form) {
    const currentUsers = Array.isArray(users) ? users.slice() : [];
    const validation = validateRegistration(form || {});
    if (!validation.isValid) {
      return {
        ok: false,
        users: currentUsers,
        message: validation.errors[0],
      };
    }

    const normalized = validation.normalized;
    const exists = currentUsers.some(function (user) {
      return normalizeEmail(user.email) === normalized.email;
    });
    if (exists) {
      return {
        ok: false,
        users: currentUsers,
        message: "An account with this email already exists.",
      };
    }

    const nextUser = {
      fullName: normalized.fullName,
      email: normalized.email,
      city: normalized.city,
      password: normalized.password,
    };

    return {
      ok: true,
      users: currentUsers.concat([nextUser]),
      user: nextUser,
      message: "Registration saved locally. You can log in now.",
    };
  }

  function loginUser(users, form) {
    const email = normalizeEmail(form.email);
    const password = String(form.password || "");
    const currentUsers = Array.isArray(users) ? users : [];
    const user = currentUsers.find(function (item) {
      return normalizeEmail(item.email) === email && String(item.password || "") === password;
    });

    if (!user) {
      return {
        ok: false,
        message: "Incorrect email or password.",
      };
    }

    return {
      ok: true,
      user: user,
      message: "Login successful. Local demo session ready.",
    };
  }

  return {
    createSeedUsers: createSeedUsers,
    evaluateScientificExpression: evaluateScientificExpression,
    loginUser: loginUser,
    registerUser: registerUser,
    validateRegistration: validateRegistration,
  };
});
