#pragma once

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include "bakkesmod/plugin/bakkesmodplugin.h"

// General Utils
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include "utils/stringify.h"
//#include "utils/threading.h"
//#include "utils/exception_safety.h"

// ImGui
#include "imgui/imgui.h"
#include "fmt/include/fmt/core.h"
#include "fmt/include/fmt/ranges.h"
#include "ImGui/imgui_searchablecombo.h"
#include "ImGui/imgui_rangeslider.h"
#include "ImGui/imgui_additions.h"
//#include "ImGui/imgui_stdlib.h" //just demo code

extern std::shared_ptr<CVarManagerWrapper> _globalCvarManager;

//template<typename S, typename... Args>
//void LOG(const S& format_str, Args&&... args)
//{
//	_globalCvarManager->log(fmt::format(format_str, args...));
//}