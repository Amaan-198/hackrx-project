const test = require("node:test");
const assert = require("node:assert/strict");

const TechLab = require("../assets/tech-lab.js");

test("evaluateScientificExpression handles arithmetic", function () {
  assert.equal(TechLab.evaluateScientificExpression("2 + 3 * 4"), 14);
});

test("evaluateScientificExpression handles scientific helpers", function () {
  assert.equal(TechLab.evaluateScientificExpression("sqrt(81) + pow(2,3)"), 17);
  assert.equal(TechLab.evaluateScientificExpression("sin(PI / 2)"), 1);
});

test("evaluateScientificExpression rejects unsupported identifiers", function () {
  assert.throws(function () {
    TechLab.evaluateScientificExpression("alert(1)");
  }, /Unsupported function/);
});

test("validateRegistration catches bad input", function () {
  const result = TechLab.validateRegistration({
    fullName: "",
    email: "bad-email",
    city: "",
    password: "short",
    confirmPassword: "mismatch",
  });

  assert.equal(result.isValid, false);
  assert.ok(result.errors.length >= 3);
});

test("registerUser adds a new local account", function () {
  const result = TechLab.registerUser(TechLab.createSeedUsers(), {
    fullName: "Asha Rao",
    email: "asha@example.com",
    city: "Mumbai",
    password: "StrongPass1",
    confirmPassword: "StrongPass1",
  });

  assert.equal(result.ok, true);
  assert.equal(result.users.length, 2);
  assert.equal(result.user.email, "asha@example.com");
});

test("loginUser accepts the seeded demo account", function () {
  const result = TechLab.loginUser(TechLab.createSeedUsers(), {
    email: "demo@decision.local",
    password: "Demo1234!",
  });

  assert.equal(result.ok, true);
  assert.equal(result.user.fullName, "Demo Analyst");
});
